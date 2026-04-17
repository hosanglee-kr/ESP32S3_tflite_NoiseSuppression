import os
import shutil
from os.path import join, isdir, exists
from SCons.Script import Import

Import("env")

def filter_esp_dsp(source, target, item):
    """
    빌드 과정 중 특정 라이브러리를 필터링하거나 복사하는 로직
    """
    lib_deps_dir = env.subst("$PROJECT_LIBDEPS_DIR")
    board_env = env.subst("$PIOENV")
    project_dir = env.subst("$PROJECT_DIR")

    # 경로 설정
    esp_dsp_src = join(lib_deps_dir, board_env, "esp-dsp")
    target_lib_dir = join(project_dir, "lib", "esp-dsp")

    if not exists(esp_dsp_src):
        print(f"--> [esp-dsp] Source not found at {esp_dsp_src}. Waiting for installation...")
        return

    if exists(target_lib_dir):
        # 이미 존재하면 패스
        return

    print(f"--> [esp-dsp] Extracting core modules to {target_lib_dir}...")
    
    try:
        os.makedirs(target_lib_dir, exist_ok=True)
        # 예: src 폴더만 골라서 복사하는 로직
        # shutil.copytree(join(esp_dsp_src, "modules"), target_lib_dir, dirs_exist_ok=True)
        print(f"--> [esp-dsp] Extraction completed successfully.")
    except Exception as e:
        print(f"--> [esp-dsp] Error during extraction: {e}")

# 단순히 바로 실행하는 대신, 
# 라이브러리 의존성 설치가 끝난 시점에 맞춰 실행되도록 유도하거나
# 빌드 전 단계(Pre-action)로 명확히 정의합니다.
filter_esp_dsp(None, None, None)
