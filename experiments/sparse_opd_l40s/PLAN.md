/goal

# Obiettivo

Configura da zero questa macchina con 4× NVIDIA L40S e realizza una pipeline end-to-end per testare **Sparse On-Policy Distillation**, partendo dal branch `odp` di Palingenesis.

Non limitarti a scrivere un piano: installa dipendenze, scarica i modelli, modifica l’harness, crea dataset e test, avvia i servizi, esegui smoke test e pilot sperimentale, raccogli metriche e produci un report finale riproducibile.

Lavora autonomamente. Non chiedere conferme per decisioni tecniche reversibili. Non cambiare silenziosamente modello, loss o precisione: ogni fallback deve essere registrato.

---

# Ipotesi sperimentale

Vogliamo verificare se supervisionare lo student solo a intervalli regolari possa ottenere un rapporto qualità/costo migliore rispetto alla supervisione token-per-token.

Confrontare:

* tutoring denso: supervisione a ogni token;
* tutoring ogni 8 token;
* tutoring ogni 32 token;
* tutoring ogni 64 token;
* tutoring ogni 128 token;
* tutoring solo alla fine;
* student base senza training;
* offline distillation/SFT come baseline.

L’ipotesi è che esista un punto intermedio, probabilmente tra 32 e 64 token, nel quale lo student conserva maggiore autonomia, evita imitazione pedissequa e utilizza meno compute del teacher senza perdere qualità.

---

# Decisioni tecniche vincolanti

## Hardware

Macchina con:

* 4× NVIDIA L40S da 48 GB;
* GPU collegate tramite PCIe;
* sistema operativo Linux appena installato;
* nessun ambiente software preesistente da considerare affidabile.

Prima di assegnare gli indici GPU esegui:

```bash
nvidia-smi
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id --format=csv
```

Scegli come coppia del teacher le due GPU con il collegamento PCIe/P2P migliore.

Salva la scelta in:

```bash
env/gpu_map.env
```

con:

```bash
TEACHER_GPUS=<gpu_a>,<gpu_b>
TRAIN_GPU=<gpu_c>
ROLLOUT_GPU=<gpu_d>
```

## Modelli

Teacher:

```text
Qwen/Qwen3-Coder-Next-FP8
```

Specifiche operative:

* 80B parametri totali;
* circa 3B attivati per token;
* checkpoint FP8 ufficiale;
* inference con SGLang;
* tensor parallel size 2;
* frozen;
* nessun gradiente;
* contesto iniziale limitato a 8192 token;
* KV cache non quantizzata durante gli esperimenti scientifici.

Student:

```text
Qwen/Qwen3-4B
```

Specifiche operative:

* BF16;
* text-only;
* non-thinking mode;
* LoRA per il primo ciclo sperimentale;
* training su una GPU;
* replica rollout su una GPU;
* vocabolario condiviso con il teacher;
* nessuna quantizzazione dello student durante il training.

Non usare `Qwen/Qwen3.5-2B` per la KL token-level: il suo vocabolario non è compatibile con quello del teacher.

## Quantizzazione

Teacher:

* utilizzare esclusivamente il checkpoint ufficiale FP8;
* non aggiungere AWQ, GPTQ, GGUF o bitsandbytes;
* non specificare `--quantization fp8` se il checkpoint viene riconosciuto automaticamente come FP8;
* verificare dai log che i pesi FP8 siano effettivamente caricati.

Student:

* pesi base BF16;
* LoRA BF16;
* optimizer state solo sugli adapter;
* niente QLoRA, INT8 o NF4, per evitare una variabile sperimentale aggiuntiva.

## Motori

Teacher:

```text
SGLang >= 0.5.8
```

Student training:

```text
PyTorch + Transformers + PEFT, integrati in Palingenesis
```

Student rollout iniziale:

```text
worker Transformers dedicato su GPU separata
```

Non usare immediatamente un secondo server SGLang per lo student. Prima costruisci una pipeline corretta con sincronizzazione NCCL degli adapter. Aggiungi un backend SGLang/vLLM per lo student solo se il profiling mostra che la generazione dello student occupa oltre il 20% del wall-clock.

---

# Layout della macchina

Usa come root:

```bash
/opt/sparse-opd
```

Se `/opt` non è scrivibile, usa:

```bash
$HOME/sparse-opd
```

Crea:

```text
sparse-opd/
├── env/
├── models/
├── cache/
│   ├── huggingface/
│   ├── torch/
│   └── datasets/
├── repos/
│   └── palingenesis/
├── venvs/
│   ├── teacher/
│   └── train/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── runs/
│   ├── smoke/
│   ├── pilot/
│   └── sweep/
├── results/
├── reports/
├── logs/
├── ops/
└── STATUS.md
```

Mantieni separati l’ambiente SGLang e quello di training, per evitare conflitti tra versioni di Torch, Transformers e CUDA.

---

# Checklist end-to-end

## 1. Preflight della macchina

* [ ] Rilevare distribuzione Linux e versione del kernel.
* [ ] Verificare driver NVIDIA e visibilità di tutte le quattro GPU.
* [ ] Eseguire un test CUDA minimo su ogni GPU.
* [ ] Verificare P2P tra tutte le coppie GPU.
* [ ] Verificare almeno 220 GB di spazio libero; raccomandati almeno 350 GB.
* [ ] Verificare almeno 64 GB di RAM; registrare la quantità disponibile.
* [ ] Verificare dimensione di `/dev/shm`.
* [ ] Verificare che nessun altro processo occupi le GPU.
* [ ] Salvare output completo in `reports/machine_inventory.md`.
* [ ] Creare `env/gpu_map.env`.
* [ ] Non aggiornare o sostituire il driver NVIDIA se CUDA funziona già.

Installa, se mancanti:

```text
git
git-lfs
curl
wget
jq
tmux
htop
nvtop
build-essential
python3.11
python3.11-dev
python3.11-venv
docker
docker-compose-plugin
```

Installa `uv` e registra la versione.

Imposta:

```bash
HF_HOME=<root>/cache/huggingface
HF_DATASETS_CACHE=<root>/cache/datasets
TORCHINDUCTOR_CACHE_DIR=<root>/cache/torch
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TOKENIZERS_PARALLELISM=true
```

Scrivi queste variabili in `env/common.env`.

## 2. Repository e branch di lavoro

* [ ] Clonare `https://github.com/mii-llm/palingenesis.git`.
* [ ] Fare checkout del branch `odp`.
* [ ] Salvare SHA iniziale in `reports/revisions.md`.
* [ ] Creare branch locale:

```text
experiment/sparse-opd-l40s
```

* [ ] Leggere `AGENTS.md`, `README.md`, `docs/on_policy_distillation.md`, `pyproject.toml` e i test OPD.
* [ ] Eseguire la test suite originale prima di modificare il codice.
* [ ] Non correggere test esistenti abbassandone la severità.
* [ ] Creare un commit separato per ogni fase funzionale.

Installazione harness:

```bash
uv venv --python 3.11 <root>/venvs/train
source <root>/venvs/train/bin/activate
uv pip install -e ".[dev]"
uv pip install peft httpx aiohttp pytest-timeout
```

Se il pin di Torch nel repository è incompatibile con il driver installato, risolvi scegliendo una build ufficiale CUDA compatibile, registra la variazione e riesegui tutta la suite.

## 3. Ambiente SGLang

Crea ambiente separato:

```bash
uv venv --python 3.11 <root>/venvs/teacher
source <root>/venvs/teacher/bin/activate
uv pip install "sglang[all]>=0.5.8"
```

Dopo l’installazione:

* [ ] Salvare `pip freeze`.
* [ ] Salvare versione SGLang, Torch, CUDA runtime e Triton.
* [ ] Eseguire import test.
* [ ] Non usare automaticamente una nightly se la release stabile funziona.
* [ ] Se serve una nightly, registrarne commit e motivazione.

## 4. Download dei modelli

Scaricare tramite `huggingface_hub` con resume abilitato:

```text
Qwen/Qwen3-Coder-Next-FP8
Qwen/Qwen3-4B
```

* [ ] Salvare revision/commit SHA dei modelli.
* [ ] Verificare integrità dei file.
* [ ] Verificare che il teacher sia il checkpoint FP8 ufficiale.
* [ ] Verificare config, vocab size, EOS e special token.
* [ ] Non scaricare la variante BF16 da 80B.
* [ ] Non creare copie duplicate dei pesi.
* [ ] Usare symlink dal path `models/` verso la cache Hugging Face.

## 5. Gate di compatibilità tokenizer

Creare:

```text
scripts/check_tokenizer_compatibility.py
```

Il test deve:

* caricare entrambi i tokenizer;
* verificare `vocab_size`;
* verificare che gli ID base rappresentino lo stesso token;
* confrontare almeno 500 token scelti deterministicamente;
* testare testo inglese, italiano, Unicode, codice Python, spazi, newline e indentazione;
* testare EOS, end-of-turn e token speciali;
* confrontare tokenizzazione e decoding delle completion;
* stampare un report macchina-leggibile JSON;
* terminare con exit code non-zero in caso di incompatibilità materiale.

Probe minimi:

```text
Hello world
L'intelligenza artificiale
def fibonacci(n: int) -> int:
    return n
é à è ò ù
JSON: {"value": 42}
tabs, spaces, newline e triple backticks
```

Output:

```text
reports/tokenizer_compatibility.json
reports/tokenizer_compatibility.md
```

Questo gate deve passare prima di eseguire training.

## 6. Teacher server SGLang

Creare:

```text
ops/start_teacher.sh
ops/stop_teacher.sh
ops/teacher_status.sh
ops/benchmark_teacher.py
ops/teacher_healthcheck.py
```

Configurazione iniziale:

```bash
CUDA_VISIBLE_DEVICES=${TEACHER_GPUS} \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-Coder-Next-FP8 \
  --tp 2 \
  --host 127.0.0.1 \
  --port 30000 \
  --context-length 8192 \
  --mem-fraction-static 0.82 \
  --chunked-prefill-size 4096 \
  --enable-metrics \
  --enable-deterministic-inference
```

Prima del comando definitivo verifica i flag disponibili nella versione installata.

Non usare il tool-call parser: il teacher viene usato per scoring, non come coding agent interattivo.

Fallback ordinati in caso di OOM:

1. ridurre `--mem-fraction-static` a 0.78;
2. ridurre context length a 4096;
3. ridurre chunked prefill a 2048;
4. disabilitare CUDA graph solo se indicato dai log;
5. provare `--enable-p2p-check`.

Non cambiare modello senza registrare un kill decision.

Benchmark teacher:

* batch size: 1, 4, 8, 16, 32;
* prompt length: 256, 512, 1024, 2048 token;
* una richiesta di decode normale;
* una richiesta di scoring con logprobs;
* top-logprobs: 32, 64, 128;
* misurare TTFT, richieste/s, token/s e VRAM;
* verificare stabilità su almeno 100 richieste consecutive.

Salvare:

```text
reports/teacher_benchmark.json
reports/teacher_benchmark.md
```

## 7. Backend teacher per Palingenesis

Aggiungere un protocollo:

```python
class TeacherBackend(Protocol):
    def score_anchors(...) -> AnchorScores:
        ...
```

Implementare:

```text
LocalTransformersTeacherBackend
SGLangTeacherBackend
```

Il backend SGLang deve usare la Native API `/generate` con:

```text
input_ids
return_logprob
top_logprobs_num
token_ids_logprob
max_new_tokens=1
temperature=0
```

Non chiedere al teacher di generare una correzione testuale. Il teacher deve restituire distribuzioni di probabilità nei punti di tutoring.

Per ogni anchor:

1. costruire il prefix fino all’anchor;
2. inviare gli ID condivisi al teacher;
3. richiedere i top-K token del teacher;
4. richiedere esplicitamente le probabilità dei top-K token dello student;
5. includere EOS/end-of-turn;
6. restituire logprob e token ID;
7. verificare che tutte le probabilità siano finite.

Batchare gli anchor prefix in una singola richiesta quando possibile. Sfruttare la prefix cache di SGLang.

Aggiungere timeout, retry limitato e circuit breaker. Non ripetere indefinitamente richieste fallite.

## 8. Loss sparse reverse-KL

Aggiungere una nuova loss:

```text
sparse_anchor_rkl
```

Per ogni posizione di tutoring:

* ottenere top-K student;
* ottenere top-K teacher;
* costruire l’unione dei token;
* aggiungere EOS;
* calcolare probabilità student esatte sull’unione;
* recuperare probabilità teacher per la stessa unione;
* calcolare la massa residua di entrambi;
* rappresentare tutto ciò che resta fuori dall’unione come un singolo residual bucket.

Loss:

```text
KL sulle categorie esplicite
+
KL tra residual mass student e residual mass teacher
```

Non distribuire artificialmente la massa residua in modo uniforme sul vocabolario.

Aggiungere epsilon numerico e clamp controllati. Loggare quante probabilità vengono clippate.

La nuova loss rappresenta una KL coarse-grained sul supporto selezionato. Non chiamarla `full_kl`.

Configurazione:

```yaml
tutoring:
  mode: sparse_anchor_rkl
  interval_tokens: 32
  anchor_window_tokens: 1
  always_include_final_anchor: true
  include_eos_anchor: true
  teacher_top_k: 64
  student_top_k: 64
  residual_bucket: true
  max_anchors_per_sequence: 64
  anchor_microbatch: 32
```

Supportare:

```text
interval_tokens: 1
interval_tokens: 8
interval_tokens: 32
interval_tokens: 64
interval_tokens: 128
interval_tokens: final
```

`interval_tokens: 1` è la baseline densa della nuova loss, non la full-distribution KL originale.

Aggiungere opzionalmente:

```text
anchor_window_tokens: 4
```

ma non usarlo nel confronto principale iniziale.

## 9. Worker rollout su GPU dedicata

GPU rollout:

* caricare Qwen3-4B BF16;
* applicare gli stessi adapter LoRA del trainer;
* generare in `torch.inference_mode()`;
* usare KV cache;
* usare Flash Attention quando supportato;
* impostare non-thinking mode;
* restituire token ID, non soltanto testo;
* eseguire un forward no-grad sulle completion per ottenere i top-K student agli anchor.

Parametri iniziali:

```yaml
sampling:
  temperature: 1.0
  top_p: 1.0
  top_k: -1
  group_size: 1
  max_new_tokens: 384
  batch_prompts: 4
```

Sincronizzazione:

* trainer su GPU training;
* rollout worker su GPU rollout;
* base model identico e frozen;
* sincronizzare solo adapter LoRA;
* preferire broadcast NCCL;
* aggiornare gli adapter dopo ogni optimizer step;
* incrementare `policy_version`;
* ogni rollout deve riportare `policy_version`;
* il trainer deve rifiutare dati prodotti da una versione diversa da quella attesa.

Aggiungere metriche:

```text
rollout/policy_version
rollout/staleness
rollout/sync_ms
rollout/generation_ms
rollout/student_topk_ms
```

Il primo esperimento scientifico deve avere staleness pari a zero.

Fallback di correttezza:

Se il worker separato non è stabile, eseguire temporaneamente rollout e training sequenzialmente sulla GPU training. Non accettare staleness non misurata.

## 10. Configurazione LoRA

Usare PEFT.

Configurazione iniziale:

```yaml
adapter:
  type: lora
  rank: 32
  alpha: 64
  dropout: 0.0
  bias: none
```

Applicare LoRA alle proiezioni lineari di attenzione e MLP effettivamente presenti in Qwen3-4B. Non assumere i nomi: ispezionare il modello e salvare la lista definitiva in configurazione.

Non usare QLoRA.

Optimizer iniziale:

```yaml
train:
  optimizer: adamw
  learning_rate: 5.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.05
  max_grad_norm: 1.0
  bf16: true
  gradient_checkpointing: true
  gradient_accumulation_steps: 4
```

Prima dello sweep sugli intervalli, eseguire un mini-sweep LR su una sola condizione:

```text
2e-5
5e-5
1e-4
```

Scegliere il valore più stabile e congelarlo per tutte le condizioni.

## 11. Dataset verificabile di programmazione

Usare come primo caso di test task Python a funzione singola con test automatici.

Preparare:

* MBPP sanitized o dataset equivalente disponibile tramite Hugging Face;
* split ufficiali quando disponibili;
* nessun esempio del test split nel training;
* deduplicazione testuale e per hash;
* prompt senza soluzione;
* test mantenuti separati dal prompt;
* seed fisso.

Creare formato JSONL:

```json
{
  "id": "task_id",
  "prompt": "descrizione del problema",
  "entry_point": "nome_funzione",
  "tests": ["assert ..."],
  "split": "train",
  "source": "mbpp_sanitized",
  "prompt_hash": "..."
}
```

Smoke split:

```text
64 train
32 dev
32 test
```

Pilot split:

* utilizzare tutti i task validi disponibili;
* preservare test split;
* limitare inizialmente il train set a un massimo di 700 task;
* creare manifest con hash e provenienza.

Aggiungere opzionalmente HumanEval+/EvalPlus solo come valutazione out-of-distribution. Non usarlo per scegliere iperparametri.

## 12. Sandbox per il codice generato

Non eseguire codice generato direttamente sull’host.

Creare immagine Docker minimale Python e runner con:

```text
network disabled
filesystem read-only
capabilities dropped
no-new-privileges
memory limit 512 MB
CPU limit 1
PID limit 128
timeout 10 secondi
tmpfs limitata
```

Comando equivalente:

```bash
docker run \
  --rm \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 128 \
  --memory 512m \
  --cpus 1 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  sparse-opd-executor
```

Catturare:

```text
syntax success
runtime success
test pass/fail
timeout
exception type
stdout/stderr limitati
```

## 13. Baseline sperimentali

Implementare condizioni:

```text
base_student
offline_teacher_sft
dense_anchor_rkl_k64_interval1
sparse_anchor_rkl_k64_interval8
sparse_anchor_rkl_k64_interval32
sparse_anchor_rkl_k64_interval64
sparse_anchor_rkl_k64_interval128
final_only_rkl
```

`offline_teacher_sft`:

* generare una soluzione teacher per ogni prompt una volta;
* salvare output e revision del teacher;
* addestrare lo student sugli output;
* non rigenerare il dataset durante il training.

Mantenere costanti tra le condizioni:

* dataset;
* split;
* seed;
* prompt template;
* max generation length;
* student training tokens;
* numero di optimizer step;
* LR;
* LoRA;
* batch globale;
* evaluation set.

Il numero di anchor e il compute teacher devono variare naturalmente: sono la variabile sperimentale.

## 14. Due modalità di confronto

### A. Budget student costante

Stesso numero di:

* optimizer step;
* prompt;
* token generati dallo student.

Misurare quanto teacher compute viene risparmiato da ogni intervallo.

### B. Budget teacher costante

Assegnare a ogni condizione lo stesso numero totale di anchor scored.

Le configurazioni sparse possono quindi vedere più prompt o più step.

Questo serve a distinguere:

* risparmio puro;
* maggiore efficienza per unità di teacher compute.

## 15. Smoke test

Prima del pilot:

* [ ] Tutti i test originali Palingenesis passano.
* [ ] Tutti i nuovi unit test passano.
* [ ] Teacher server risponde.
* [ ] Teacher restituisce logprob finite.
* [ ] Student rollout genera codice.
* [ ] Adapter sync produce hash identici sulle due GPU.
* [ ] Staleness uguale a zero.
* [ ] Un training step completa forward, backward e update.
* [ ] Dieci training step consecutivi senza leak di memoria.
* [ ] Checkpoint e resume funzionano.
* [ ] Sandbox esegue i test.
* [ ] Metriche vengono scritte.
* [ ] Nessun OOM.
* [ ] Nessun processo zombie.

Eseguire un run da 20 step per:

```text
interval1
interval32
interval128
final
```

Non passare al pilot finché questi gate non sono verdi.

## 16. Unit test obbligatori

Aggiungere test per:

* anchor selection corretta;
* intervallo 1;
* intervallo 32;
* final-only;
* EOS sempre incluso;
* sequence più corta dell’intervallo;
* completion vuota;
* truncation;
* identiche distribuzioni → KL circa zero;
* distribuzioni differenti → KL positiva;
* residual bucket;
* probabilità finite;
* gradienti non nulli;
* gradienti senza NaN;
* teacher timeout;
* retry;
* response malformed;
* mismatch tokenizer;
* mismatch policy version;
* adapter synchronization;
* checkpoint/resume;
* determinismo del dataset;
* sandbox timeout.

## 17. Pilot sperimentale

Dopo gli smoke test:

```text
300 optimizer step per condizione
1 seed iniziale
max_new_tokens 384
teacher_top_k 64
student_top_k 64
```

Condizioni iniziali:

```text
base
offline_sft
interval1
interval32
interval64
interval128
final
```

Dopo il pilot:

* scegliere le migliori due condizioni sparse;
* ripeterle con seed 0, 1 e 2;
* ripetere dense e offline SFT con gli stessi seed;
* eseguire ablation top-K 64 contro 128 sulla migliore condizione;
* eseguire una piccola ablation `anchor_window_tokens: 1` contro `4`.

Non eseguire tutte le combinazioni prima del pilot.

## 18. Metriche

Qualità:

```text
pass@1
test pass rate
syntax success rate
runtime success rate
task completion rate
mean tests passed per task
```

Training:

```text
loss
anchor KL
residual student mass
residual teacher mass
gradient norm
learning rate
completion length
output entropy
```

Efficienza:

```text
teacher requests
teacher anchor positions
teacher scored tokens
student generated tokens
teacher wall-clock
student rollout wall-clock
backward wall-clock
total wall-clock
GPU-hours per GPU
peak VRAM
energy estimate se disponibile
```

Autonomia/diversità:

```text
normalized AST uniqueness
exact duplicate rate
solution length
edit distance rispetto al teacher
teacher/student agreement
```

Sincronizzazione:

```text
policy staleness
adapter sync latency
policy version mismatch count
```

Calcolare soprattutto:

```text
pass@1 per teacher GPU-hour
pass@1 per milione di anchor scored
miglioramento rispetto al base
miglioramento rispetto a offline SFT
```

## 19. Profiling e ottimizzazione

Usare profiling solo dopo la correttezza.

Misurare percentuale di wall-clock per:

```text
student rollout
student top-K pass
teacher scoring
student gradient forward
backward
adapter synchronization
sandbox evaluation
```

Ottimizzazioni ammesse:

* aumentare teacher anchor batch;
* aumentare prefix-cache hit rate;
* pipeline asincrona tra rollout e teacher scoring;
* pinned CPU memory;
* NCCL adapter broadcast;
* prefetch del batch successivo;
* ridurre serializzazione JSON;
* usare input token IDs;
* usare connessioni HTTP persistenti;
* compilare soltanto componenti stabili.

Non introdurre staleness dello student per aumentare throughput senza una specifica ablation.

Se student rollout supera il 20% del tempo:

* valutare SGLang/vLLM per la replica rollout;
* usare hot update degli adapter solo se supportato in modo stabile;
* verificare policy version dopo ogni update;
* confrontare output e logprob contro Transformers;
* mantenere il backend Transformers come riferimento.

## 20. Kill switch e fallback

Interrompere il run e produrre diagnosi se:

* tokenizer incompatibile;
* KL NaN o infinita;
* residual student mass media sopra 0,20;
* policy staleness diversa da zero nel run principale;
* teacher logprobs cambiano materialmente tra batch size diversi;
* più del 5% delle richieste teacher fallisce;
* VRAM cresce continuamente per oltre 50 step;
* pass@1 crolla oltre il 30% rispetto al base;
* sandbox non isola correttamente il codice;
* checkpoint resume non riproduce lo step successivo.

Se Coder-Next FP8 non entra su due L40S:

1. ridurre context e memoria KV;
2. verificare che sia realmente caricato FP8;
3. verificare topology e P2P;
4. provare un’altra coppia GPU;
5. solo dopo documentare un fallback a un teacher Qwen più piccolo con vocabolario compatibile.

Non usare CPU offload nel run principale: altera troppo il profilo prestazionale.

## 21. Configurazioni da creare

Creare almeno:

```text
configs/sparse_opd/base.yaml
configs/sparse_opd/offline_sft.yaml
configs/sparse_opd/interval_1.yaml
configs/sparse_opd/interval_8.yaml
configs/sparse_opd/interval_32.yaml
configs/sparse_opd/interval_64.yaml
configs/sparse_opd/interval_128.yaml
configs/sparse_opd/final_only.yaml
configs/sparse_opd/smoke.yaml
```

Creare uno script che generi configurazioni derivate senza duplicare tutti i campi.

Ogni run directory deve contenere una copia immutabile della configurazione risolta.

## 22. Script operativi

Creare:

```text
ops/bootstrap_machine.sh
ops/create_envs.sh
ops/download_models.sh
ops/start_teacher.sh
ops/stop_teacher.sh
ops/teacher_status.sh
ops/run_existing_tests.sh
ops/run_smoke.sh
ops/run_pilot.sh
ops/run_sweep.sh
ops/resume_run.sh
ops/evaluate_checkpoint.sh
ops/collect_results.py
ops/make_report.py
```

Tutti gli script devono:

* usare `set -euo pipefail`;
* stampare timestamp;
* salvare log;
* poter essere rilanciati;
* non riscaricare file validi;
* non cancellare risultati precedenti;
* supportare `--dry-run` quando sensato.

## 23. Tracking e riproducibilità

Usare logging locale come fonte primaria.

W&B può essere opzionale, ma il run non deve dipendere da una connessione esterna.

Ogni run deve salvare:

```text
git SHA harness
git diff
model revision
dataset manifest
pip freeze teacher
pip freeze trainer
nvidia-smi
GPU topology
config resolved
seed
start/end timestamp
checkpoint
metriche JSONL
stdout/stderr
```

Usare nomi run:

```text
<date>_<condition>_seed<seed>_<shortsha>
```

Aggiornare continuamente:

```text
STATUS.md
```

con checkbox, risultati e problemi aperti.

## 24. Report finale

Produrre:

```text
reports/final_report.md
results/summary.csv
results/summary.json
results/pareto_quality_cost.csv
```

Il report deve includere:

1. configurazione hardware;
2. versioni software;
3. architettura implementata;
4. differenza tra full KL originale e sparse anchor RKL;
5. compatibilità tokenizer;
6. benchmark teacher;
7. risultati smoke;
8. risultati pilot;
9. qualità per condizione;
10. teacher compute per condizione;
11. wall-clock;
12. GPU-hours;
13. Pareto qualità/costo;
14. eventuale punto ottimo della frequenza;
15. errori e limitazioni;
16. anomalie;
17. raccomandazione sul prossimo esperimento.

Creare almeno questi grafici:

```text
pass@1 vs intervallo tutoring
pass@1 vs teacher GPU-hours
pass@1 vs anchor scored
wall-clock breakdown
residual mass vs step
KL vs step
completion length vs step
```

Non usare soltanto la training loss per dichiarare un miglioramento.

## 25. Criteri di completamento

Il goal è completato soltanto quando:

* [ ] La macchina è configurata.
* [ ] Il teacher gira su due L40S.
* [ ] Lo student training gira su una L40S.
* [ ] Il rollout worker gira sulla quarta L40S.
* [ ] I tokenizer risultano compatibili.
* [ ] La nuova sparse anchor RKL è implementata.
* [ ] Tutti i test originali passano.
* [ ] Tutti i nuovi test passano.
* [ ] Gli smoke run sono completati.
* [ ] Almeno un pilot end-to-end è completato.
* [ ] Sono disponibili metriche di qualità e costo.
* [ ] È stato prodotto il report finale.
* [ ] Sono riportati i comandi esatti per riprendere il lavoro.

Alla fine stampa un riepilogo conciso con:

```text
STATO
GPU MAP
TEACHER ENDPOINT
MODEL REVISION
HARNESS COMMIT
TEST PASSATI
RUN COMPLETATI
RISULTATO PRINCIPALE
POSIZIONE DEI REPORT
COMANDO PER RIPRENDERE
```

Non fermarti alla preparazione dell’ambiente: esegui realmente smoke test e pilot, compatibilmente con le risorse disponibili.
