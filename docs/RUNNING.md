# Running the Project

This document gives the recommended command sequence for running the BEP pipeline from a clean local checkout.

Run all commands from the repository root.

---

## 1. Environment Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check Python can import the main libraries:

```powershell
python -c "import pandas, torch, transformers; print('imports ok')"
```

Check whether PyTorch sees the GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

If CUDA is unavailable, the placeholder/debug modes can still be useful. The Transformers Qwen path is not recommended on CPU.

---

## 2. Confirm Local Artifacts

Before running the real pipeline, make sure this folder exists in the repository root:

```text
Automated Deception Classifier (Projectfolder)/
```

Minimum required files for baseline and pool construction:

```text
Automated Deception Classifier (Projectfolder)/hippocorpus_training_truncated.csv
Automated Deception Classifier (Projectfolder)/hippocorpus_test_truncated.csv
Automated Deception Classifier (Projectfolder)/DistilBERT/
```

Minimum required files for validity checking:

```text
Automated Deception Classifier (Projectfolder)/local_models/all-MiniLM-L6-v2/
```

See `docs/LOCAL_ARTIFACTS.md` for the full expected structure.

---

## 3. Freeze Missing Local Models

Only run these if the local model folders are missing.

### SBERT validity model

```powershell
python Scripts_code\freeze_sbert_model.py
```

### Qwen Transformers paraphraser model

```powershell
python Scripts_code\freeze_local_llm_model.py
```

The GGUF model for llama.cpp is handled manually and is not downloaded by these scripts.

---

## 4. Run Baseline Inference

```powershell
python Scripts_code\01_baseline_inference.py
```

Expected output folder:

```text
Scripts_code/outputs/
```

Expected split-specific outputs:

```text
baseline_predictions_train_dev.csv
candidate_pool_train_dev.csv
baseline_summary_train_dev.json
baseline_predictions_test_final.csv
candidate_pool_test_final.csv
baseline_summary_test_final.json
```

The candidate pools contain only examples that the fixed victim classifier originally classified correctly.

---

## 5. Freeze Pilot and Main Attack Pools

```powershell
python Scripts_code\02_make_attack_pool.py
```

Expected outputs:

```text
candidate_pool_pilot_20.csv
candidate_pool_main_160.csv
candidate_pool_freeze_meta.json
```

The script refuses to overwrite existing frozen pools unless `--overwrite` is used.

Only use:

```powershell
python Scripts_code\02_make_attack_pool.py --overwrite
```

when you intentionally want to regenerate the frozen attack pools.

---

## 6. Run a Placeholder Pilot Smoke Test

This tests the full attack loop structure without using a real LLM.

```powershell
python Scripts_code\05_attack_pilot.py `
  --paraphraser placeholder `
  --max-examples 2 `
  --max-generation-attempts 3 `
  --max-classifier-queries 3 `
  --overwrite
```

Expected outputs:

```text
attack_attempts_pilot_placeholder.csv
attack_results_pilot_placeholder.csv
attack_pilot_meta_placeholder.json
```

---

## 7. Run Invalid-Debug Pilot Smoke Test

This tests that invalid candidates are blocked before classifier querying.

```powershell
python Scripts_code\05_attack_pilot.py `
  --paraphraser invalid_debug `
  --max-examples 2 `
  --max-generation-attempts 3 `
  --max-classifier-queries 3 `
  --overwrite
```

Expected outputs:

```text
attack_attempts_pilot_invalid_debug.csv
attack_results_pilot_invalid_debug.csv
attack_pilot_meta_invalid_debug.json
```

Important expected behavior:

- invalid candidates are logged;
- invalid candidates are not queried against the classifier;
- the metadata should report that integrity checks passed.

---

## 8. Run Local LLM Mode with Transformers Backend

This path loads the local Qwen Transformers model directly in Python.

```powershell
python Scripts_code\05_attack_pilot.py `
  --paraphraser local_llm `
  --local-llm-backend transformers `
  --max-examples 1 `
  --max-generation-attempts 2 `
  --max-classifier-queries 2 `
  --local-llm-max-new-tokens 120 `
  --overwrite
```

This backend is useful as a fallback and reproducibility reference, but it is much slower than the llama.cpp server backend on the current hardware.

---

## 9. Run Local LLM Mode with llama.cpp Server Backend

The llama.cpp server should run locally on:

```text
http://127.0.0.1:8080/v1
```

Start the server in a separate PowerShell window. Example:

```powershell
$server = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
$model = "Automated Deception Classifier (Projectfolder)\local_models\Qwen3-4B-Instruct-2507-GGUF\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
& $server -m $model --host 127.0.0.1 --port 8080 --ctx-size 4096 -ngl 999
```

In another PowerShell window, verify the server is reachable:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/models" -TimeoutSec 10
```

Then run a small diagnostic:

```powershell
python Scripts_code\05_attack_pilot.py `
  --paraphraser local_llm `
  --local-llm-backend llama_cpp_server `
  --local-llm-server-url "http://127.0.0.1:8080/v1" `
  --local-llm-server-model "Qwen3-4B-Instruct-2507-Q4_K_M.gguf" `
  --local-llm-gguf-model-path "Automated Deception Classifier (Projectfolder)\local_models\Qwen3-4B-Instruct-2507-GGUF\Qwen3-4B-Instruct-2507-Q4_K_M.gguf" `
  --local-llm-context-size 4096 `
  --local-llm-gpu-layers 999 `
  --max-examples 5 `
  --max-generation-attempts 3 `
  --max-classifier-queries 3 `
  --local-llm-max-new-tokens 160 `
  --local-llm-temperature 0.45 `
  --local-llm-top-p 0.90 `
  --overwrite
```

The server is local-only when bound to `127.0.0.1`. It does not send text to a cloud provider.

To stop the server:

```powershell
Get-Process | Where-Object { $_.ProcessName -like "llama*" } | Stop-Process -Force
```

To verify it stopped:

```powershell
Get-Process | Where-Object { $_.ProcessName -like "llama*" }
Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue
```

---

## 10. Runtime Diagnostics

The pilot script can print local runtime information:

```powershell
python Scripts_code\05_attack_pilot.py --print-runtime-info
```

This is useful for checking:

- PyTorch version;
- CUDA availability;
- GPU name;
- selected local LLM backend;
- configured local model paths;
- configured server URL.

---

## 11. Output Interpretation

For each pilot run, the script writes three files:

```text
attack_attempts_pilot_<mode>.csv
attack_results_pilot_<mode>.csv
attack_pilot_meta_<mode>.json
```

Attempt-level rows record each generated candidate, validity status, failed checks, classifier-query status, prediction, and generation timing.

Result-level rows summarize one attacked item under one feedback condition.

Metadata records configuration, success definition, model/backend settings, prompt hash, output paths, and integrity checks.

---

## 12. Before Running a Full Pilot

Before running a full pilot, confirm:

- the frozen pilot pool exists;
- the local SBERT folder exists;
- the DistilBERT victim folder exists;
- placeholder and invalid-debug runs pass;
- the local LLM diagnostic produces valid paraphrases;
- generation settings are frozen and documented;
- validity thresholds are not changed differently across feedback conditions.
