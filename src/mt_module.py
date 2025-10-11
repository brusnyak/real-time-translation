import torch
from transformers import MarianMTModel, MarianTokenizer
import time


class MTModule:
    def __init__(self, source_lang="en", target_lang="sk", device="cpu"):
        """
        Initialize MT module optimized for M1 Mac.
        Removes ONNX complexity - it doesn't help on your hardware.
        
        Args:
            source_lang: Source language code (e.g., "en")
            target_lang: Target language code (e.g., "sk")
            device: Device to use ("cpu", "mps", or "cuda")
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.device = device
        self.model_name = "Helsinki-NLP/opus-mt-en-sk"
        
        print(f"[MT] Initializing MarianMT: {source_lang} -> {target_lang}")
        print(f"[MT] Device: {self.device}")
        
        start_time = time.time()
        self._init_model()
        self._load_time = time.time() - start_time
        
        print(f"[MT] Model initialized in {self._load_time:.3f}s")
        print(f"[MT] Warm up first call takes ~3.8s, subsequent calls ~0.6-0.8s")
    
    def _init_model(self):
        """Initialize the translation model."""
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)
        
        # Move to device
        if self.device == "mps" and torch.backends.mps.is_available():
            self.model = self.model.to("mps")
            print("[MT] Model loaded on MPS (Metal Performance Shaders)")
        elif self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("[MT] Model loaded on CUDA")
        else:
            self.device = "cpu"
            self.model = self.model.to("cpu")
            print("[MT] Model loaded on CPU")
        
        self.model.eval()
    
    def translate(self, text, max_new_tokens=128, num_beams=1):
        """
        Translate text from source to target language.
        
        Args:
            text: Text to translate
            max_new_tokens: Maximum length of generated translation
            num_beams: Beam search width (1 = greedy, faster)
        
        Returns:
            Translated text
        """
        start_time = time.time()
        
        # Tokenize
        encoded_text = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        encoded_text = {k: v.to(self.device) for k, v in encoded_text.items()}
        
        # Generate with inference mode
        with torch.inference_mode():
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **encoded_text,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=False,
                    use_cache=True
                )
        
        # Decode
        translated_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
        
        elapsed = time.time() - start_time
        print(f"[MT] Translation in {elapsed:.3f}s")
        
        return translated_text
    
    def _fix_slovak_grammar(self, text):
        """Basic post-processing for Slovak text."""
        text = text.replace("  ", " ")
        text = text.replace(" .", ".")
        text = text.replace(" ,", ",")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace(" :", ":")
        text = text.replace(" ;", ";")
        return text.strip()
    
    def benchmark(self, test_texts=None, num_runs=3):
        """Benchmark translation speed."""
        if test_texts is None:
            test_texts = [
                "Hello, how are you?",
                "In this experiment, the system converts spoken English into text, translates it into Slovak and then synthesizes it back into speech.",
                "The quick brown fox jumps over the lazy dog."
            ]
        
        print("\n" + "="*60)
        print("[MT] BENCHMARK REPORT")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Runs per text: {num_runs}\n")
        
        all_times = []
        
        for text in test_texts:
            print(f"Text: {text[:50]}...")
            print(f"  Length: {len(text)} chars")
            
            times = []
            for run in range(num_runs):
                start = time.perf_counter()
                self.translate(text)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                print(f"    Run {run + 1}: {elapsed:.3f}s")
            
            avg_time = sum(times) / len(times)
            all_times.extend(times)
            
            print(f"  Average: {avg_time:.3f}s")
            print(f"  Range: {min(times):.3f}s - {max(times):.3f}s\n")
        
        overall_avg = sum(all_times) / len(all_times)
        print("="*60)
        print(f"Overall average inference time: {overall_avg:.3f}s")
        print(f"Min: {min(all_times):.3f}s, Max: {max(all_times):.3f}s")
        print("="*60 + "\n")
        
        return all_times


if __name__ == "__main__":
    print("=== Testing on MPS ===")
    if torch.backends.mps.is_available():
        mt_mps = MTModule(device="mps")
        mt_mps.benchmark()
    else:
        print("MPS not available")