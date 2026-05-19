import torch
import time
import threading

def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)

log(f"Device: {torch.cuda.get_device_name(0)}")
log(f"VRAM totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
log(f"PyTorch: {torch.__version__}")
print()

# === CPU benchmark ===
# Usiamo fp32 e una matrice piu' piccola: la CPU non e' fatta per matmul giganti
size_cpu = 4096
log(f"Creo matrici {size_cpu}x{size_cpu} fp32 sulla CPU...")
a_cpu = torch.randn(size_cpu, size_cpu, dtype=torch.float32)
b_cpu = torch.randn(size_cpu, size_cpu, dtype=torch.float32)

log("Avvio matmul su CPU (atteso: 5-30 secondi)...")
done = threading.Event()
t0 = time.time()
def heartbeat():
    while not done.wait(3):
        log(f"  ...CPU al lavoro ({time.time()-t0:.0f}s elapsed)")
threading.Thread(target=heartbeat, daemon=True).start()

c_cpu = a_cpu @ b_cpu
done.set()
t_cpu = time.time() - t0
# Throughput CPU in GFLOPS
gflops_cpu = (2 * size_cpu**3) / t_cpu / 1e9
log(f"CPU matmul {size_cpu}x{size_cpu} fp32: {t_cpu:.2f}s --> {gflops_cpu:.1f} GFLOPS")
print()

# === GPU benchmark ===
size_gpu = 8192
log(f"Sposto matrici {size_gpu}x{size_gpu} fp16 sulla GPU...")
a_gpu = torch.randn(size_gpu, size_gpu, dtype=torch.float16, device='cuda')
b_gpu = torch.randn(size_gpu, size_gpu, dtype=torch.float16, device='cuda')
torch.cuda.synchronize()

log("Warm-up GPU (3 run a vuoto)...")
for _ in range(3):
    _ = a_gpu @ b_gpu
torch.cuda.synchronize()

log("Avvio benchmark GPU (20 run)...")
t0 = time.time()
for _ in range(20):
    c_gpu = a_gpu @ b_gpu
torch.cuda.synchronize()
t_gpu = (time.time() - t0) / 20
tflops_gpu = (2 * size_gpu**3) / t_gpu / 1e12
log(f"GPU matmul {size_gpu}x{size_gpu} fp16: {t_gpu*1000:.2f}ms per run --> {tflops_gpu:.1f} TFLOPS")
print()

# === Confronto normalizzato ===
# Normalizziamo a una matmul fp32 4096x4096 equivalente, per confronto onesto
log("=== Riassunto ===")
log(f"CPU:  {gflops_cpu:.1f} GFLOPS (fp32)")
log(f"GPU:  {tflops_gpu*1000:.0f} GFLOPS (fp16)")
log(f"Rapporto teorico GPU/CPU: ~{(tflops_gpu*1000)/gflops_cpu:.0f}x")