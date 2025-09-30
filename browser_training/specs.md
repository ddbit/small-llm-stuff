Titolo: Prototipo browser-native di classificazione testo con backbone DistilBERT congelato e head addestrabile (JS + WebGPU + WASM), con hook per Federated Learning

Obiettivo:

* Eseguire inferenza DistilBERT quantizzato nel browser (WebGPU, fallback WASM).
* Addestrare localmente solo la head di classificazione (W,b).
* Salvare pesi in IndexedDB.
* Fornire hook per invio/merge Δpesi (FL).
* Zero dipendenze native. Solo web standard.

Stack vincolante:

* JS/TS vanilla + Vite.
* WebGPU per compute; fallback onnxruntime-web WASM se WebGPU assente.
* Tokenizer JSON.
* IndexedDB per persistenza.
* Web Workers per tokenize/preprocess.
* WebRTC data channel (stub) per FL.

Deliverable richiesti:

* `index.html`, `src/main.ts`, `src/model.ts`, `src/head.ts`, `src/fl.ts`, `src/tokenizer.ts`, `src/storage.ts`, `src/ui.ts`, `public/` (modello/asset).
* Script build/dev.
* Demo con 3 label: Dev, Meeting, Email.

Dati/modello:

* Usa DistilBERT quantizzato ONNX (INT8). Inserisci placeholder URL `public/distilbert-int8.onnx` e `public/tokenizer.json`.
* Max seq len 256.

Funzioni chiave da implementare:

1. Rilevamento backend

```ts
// preferisci WebGPU, altrimenti WASM
initRuntime(): Promise<'webgpu'|'wasm'>
```

2. Tokenizzazione (Worker)

```ts
// tokenize(text: string) -> Int32Array ids
// usa tokenizer WordPiece/JSON
```

3. Inferenza backbone congelato

* Con onnxruntime-web:

```ts
// embedCLS(ids: Int32Array) -> Float32Array[768]
// run ONNX session (webgpu o wasm), estrai [CLS]
```

4. Head di classificazione addestrabile

```ts
// Parametri: W[768×3], b[3] inizializzati ~N(0,0.02)
forward(z: Float32Array) -> logits[3]
trainStep(z: Float32Array, y: 0|1|2, lr=1e-2) -> loss
softmax, crossEntropy, outer, SGD/Adam (in WASM o JS puro)
```

5. Pipeline training locale

```ts
// fitLocal(samples: Array<{text,label}>, epochs=5, batch=8)
// tokenizza in worker, embed CLS via backbone, aggiorna solo W,b
// salva periodicamente in IndexedDB
```

6. Inferenza end-to-end

```ts
predict(text: string) -> {label: 'Dev'|'Meeting'|'Email', probs: number[3]}
```

7. Persistenza

```ts
saveHead(W,b) / loadHead()
// IndexedDB + versioning {model_id, head_version}
```

8. Federated Learning hook (stub)

```ts
// getDelta(): quantizza ΔW,Δb in int8 + metadata
// applyAggregate(delta): dequantizza e applica media pesata
// flClientRound(serverUrl|webrtcPeer): invia delta, riceve aggregato
```

9. UI minimale

* Campo testo, pulsante Predict, output label+probs.
* Selettore label corretta → chiama `trainStep` con feedback.
* Bottoni: Save, Load, Send Delta, Apply Aggregate.
* Indicatori: backend attivo, latenza ms, memoria stimata.

10. Valutazione rapida

* Log: latenza inferenza, loss medio training, accuracy su mini-set locale.
* Soglia “Non determinato” se max(probs)<0.55.

File skeleton richiesti (codice completo):

* `index.html`: layout minimale, inclusione `main.ts`.
* `src/main.ts`: bootstrap, initRuntime, bind UI.
* `src/model.ts`: sessione ONNX (crea due provider: webgpu/wasm), `embedCLS`.
* `src/head.ts`: struttura W,b; operazioni tensori; optimizer; `forward`, `trainStep`.
* `src/tokenizer.ts`: loader tokenizer JSON; worker `tokenize`.
* `src/storage.ts`: IndexedDB save/load `W,b` e metadati.
* `src/fl.ts`: serializza Δpesi, quantizza/dequantizza, stub WebRTC/HTTP.
* `src/ui.ts`: handler UI, metrica, grafico probs (canvas semplice).
* `vite.config.ts`: asset loader per ONNX/JSON.

Accettazione:

* Demo gira su Chrome/Edge recenti.
* Backbone: inferenza CLS <100 ms su laptop medio (WebGPU).
* Training locale: convergenza su 20 esempi/label in ≤5 epoche.
* Persistenza/restore head funzionanti.
* Export/import Δpesi funzionanti (fake server ok).

Note implementative:

* ONNX I/O names: specifica corretta per DistilBERT (input_ids, attention_mask). Se assenti, mappa dinamicamente.
* Normalizza logits con temperature scaling opzionale.
* Evita copie superflue tra JS e GPU; batch embedding se possibile.
* Inserisci TODO per sicurezza FL: clipping, DP, secure aggregation.

Output atteso: repository con tutti i file, codice compilabile, istruzioni `npm i && npm run dev`, README con passi, limiti, estensioni future (LoRA/BitFit).
