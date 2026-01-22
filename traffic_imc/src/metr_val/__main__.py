import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 데이터 및 모델 리스트
data_list = ["KNN", "MICE", "BGCP", "TRMF", "BRITS"]
model_list = ["stgcn", "dcrnn", "agcrn", "mlcaformer", "rnn"]

# 로그 파일 경로 (traffic_imc 디렉토리 기준)
LOG_FILE = Path(__file__).parent.parent.parent / "training_report.log"
LOG_LOCK = threading.Lock()  # 로그 파일 쓰기 동기화


def run_model_standalone(
    data_name: str, model_type: str, gpu_id: int, code: int = 0, timeout: int = 10800
):
    """별도 프로세스로 모델을 실행하여 OOM 파급 효과 차단

    Args:
        data_name: 데이터 보간 방법 이름 (KNN, MICE, BGCP, TRMF, BRITS)
        model_type: 모델 타입 (stgcn, dcrnn, agcrn, mlcaformer, rnn)
        gpu_id: 사용할 GPU 장치 ID
        code: 실행 코드 번호 (WandB 로깅용)
        timeout: 타임아웃 시간(초), 기본 3시간 (10800초)

    Returns:
        tuple: (성공 여부, 에러 메시지, 소요 시간)
    """
    cmd = [
        sys.executable,
        "-m",
        f"metr_val.{model_type}",
        "--data",
        data_name,
        "--gpu",
        str(gpu_id),
        "--code",
        str(code),
    ]

    print(f">>> [GPU {gpu_id}] [실행 중] {model_type.upper()} | 데이터: {data_name}")
    start_time = time.time()

    try:
        # 프로세스 실행 및 완료 대기 (타임아웃 포함)
        # timeout이 0 이하면 타임아웃 비활성화
        run_timeout = timeout if timeout > 0 else None
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=run_timeout
        )
        elapsed = time.time() - start_time
        print(
            f"<<< [GPU {gpu_id}] [완료] {model_type.upper()} | 데이터: {data_name} | {elapsed:.1f}초"
        )
        return True, "", elapsed

    except subprocess.TimeoutExpired:
        # 타임아웃 (OOM으로 hang 상태일 가능성)
        elapsed = time.time() - start_time
        error_msg = f"Timeout after {timeout}s (likely OOM or infinite loop)"
        print(
            f"!!! [GPU {gpu_id}] [타임아웃] {model_type.upper()} | 데이터: {data_name} | {elapsed:.1f}초"
        )
        return False, error_msg, elapsed

    except subprocess.CalledProcessError as e:
        # 프로세스가 non-zero exit code로 종료 (OOM kill, 에러 등)
        elapsed = time.time() - start_time
        error_info = e.stderr if e.stderr else str(e)
        # OOM 시그널 감지 (일반적으로 exit code 137 = SIGKILL by OOM killer)
        oom_indicator = " [OOM KILLED]" if e.returncode == 137 else ""
        print(
            f"!!! [GPU {gpu_id}] [실패{oom_indicator}] {model_type.upper()} | 데이터: {data_name} | {error_info[:200]}..."
        )
        return False, error_info, elapsed

    except Exception as e:
        # 기타 예상치 못한 에러
        elapsed = time.time() - start_time
        error_msg = f"Unexpected error: {str(e)}"
        print(
            f"!!! [GPU {gpu_id}] [예외] {model_type.upper()} | 데이터: {data_name} | {error_msg}"
        )
        return False, error_msg, elapsed


def log_result(
    model_type: str,
    data_name: str,
    gpu_id: int,
    success: bool,
    elapsed: float,
    error: str = "",
):
    """학습 결과를 로그 파일에 기록 (thread-safe)"""
    status = "SUCCESS" if success else "FAILED"
    with LOG_LOCK:
        with open(LOG_FILE, "a") as log:
            log.write(
                f"{time.ctime()} | GPU{gpu_id} | {model_type:12} | {data_name:6} | {status:7} | {elapsed:.1f}s\n"
            )
            if not success and error:
                log.write(f"    ERROR: {error[:500]}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train all models with all datasets in parallel"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=8, help="Number of GPUs to use (default: 8)"
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (e.g., '0,1,2,3'). If not specified, uses 0 to num_gpus-1",
    )
    parser.add_argument(
        "--code", type=int, default=3, help="Run code for WandB logging"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run tasks sequentially on GPU 0 (ignores --num-gpus)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Timeout per task in seconds (default: 0 = no timeout, >0 = timeout in seconds)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=model_list,
        help=f"학습할 특정 모델 리스트 (기본: 전체). 선택 가능: {', '.join(model_list)}",
    )
    parser.add_argument(
        "--data-methods",
        nargs="+",
        default=None,
        choices=data_list,
        help=f"사용할 특정 데이터 보간 방법 리스트 (기본: 전체). 선택 가능: {', '.join(data_list)}",
    )
    args = parser.parse_args()

    # GPU IDs 설정
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = list(range(args.num_gpus))

    # 모델 및 데이터 선택 (None이면 전체 사용)
    active_models = args.models if args.models is not None else model_list
    active_data = args.data_methods if args.data_methods is not None else data_list
    
    tasks = [(d, m) for d in active_data for m in active_models]
    total_tasks = len(tasks)
    
    # 실행 정보 출력
    print("=" * 70)
    print(f"총 {total_tasks}개 작업 (데이터: {len(active_data)}개 × 모델: {len(active_models)}개)")
    print(f"데이터 방법: {', '.join(active_data)}")
    print(f"모델: {', '.join(active_models)}")
    if args.sequential:
        print(f"실행 모드: 순차 실행 (GPU 0)")
    else:
        print(f"실행 모드: 병렬 실행 (GPU: {gpu_ids}, 동시 실행: {len(gpu_ids)}개)")
    print(f"타임아웃: {'비활성화' if args.timeout <= 0 else f'{args.timeout}초 ({args.timeout/3600:.1f}시간)'}")
    print(f"로그 파일: {LOG_FILE}")
    print("=" * 70)

    if args.sequential:
        # 순차 실행 (기존 방식)
        for idx, (data_name, model_type) in enumerate(tasks, 1):
            print(f"\n[{idx}/{total_tasks}] ", end="")
            success, err, elapsed = run_model_standalone(
                data_name, model_type, gpu_id=0, code=args.code, timeout=args.timeout
            )
            log_result(model_type, data_name, 0, success, elapsed, err)
    else:
        # 병렬 실행 (멀티 GPU)
        completed = 0

        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            # 작업 제출: 각 작업을 라운드로빈 방식으로 GPU에 할당
            futures = {}
            for idx, (data_name, model_type) in enumerate(tasks):
                gpu_id = gpu_ids[idx % len(gpu_ids)]  # 라운드로빈 GPU 할당
                future = executor.submit(
                    run_model_standalone,
                    data_name,
                    model_type,
                    gpu_id,
                    args.code,
                    args.timeout,
                )
                futures[future] = (data_name, model_type, gpu_id)

            # 완료된 작업 처리
            for future in as_completed(futures):
                completed += 1
                data_name, model_type, gpu_id = futures[future]

                try:
                    success, err, elapsed = future.result()
                    log_result(model_type, data_name, gpu_id, success, elapsed, err)
                    print(f"[진행률: {completed}/{total_tasks}]")
                except Exception as e:
                    print(f"!!! [예외] {model_type.upper()} | {data_name} | {str(e)}")
                    log_result(model_type, data_name, gpu_id, False, 0, str(e))

    print("\n" + "=" * 70)
    print("모든 작업 완료!")
    print(f"결과 확인: {LOG_FILE}")
    print("=" * 70)
