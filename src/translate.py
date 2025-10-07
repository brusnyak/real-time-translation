from googletrans import Translator as GoogleTranslator # Alias to avoid conflict with class name

class Translator:
    def __init__(self, src_lang="en", tgt_lang="sk"):
        """
        Initializes the Google Translate API.
        :param src_lang: Source language code (e.g., "en" for English).
        :param tgt_lang: Target language code (e.g., "sk" for Slovak).
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.translator = GoogleTranslator()
        print(f"Google Translator initialized for {src_lang} to {tgt_lang}.")

    async def translate_text(self, text: str) -> str:
        """
        Translates the input text from source to target language using Google Translate.
        :param text: The text to translate.
        :return: The translated text.
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")

        # Temporary workaround for specific translation issue
        processed_text = text.replace("Slavak", "slovenčiny")
        
        try:
            result = await self.translator.translate(processed_text, src=self.src_lang, dest=self.tgt_lang)
            return result.text
        except Exception as e:
            print(f"Error during Google Translate API call: {e}")
            return text # Fallback to original text on error

if __name__ == "__main__":
    import asyncio

    async def run_examples():
        # Example usage: English to Slovak
        en_to_sk_translator = Translator(src_lang="en", tgt_lang="sk")
        english_sentence = "Hello, how are you today? I hope you are doing well. This is Slovak."
        slovak_translation = en_to_sk_translator.translate_text(english_sentence)
        print(f"English: {english_sentence}")
        print(f"Slovak: {slovak_translation}")

        print("\n" + "="*30 + "\n")

        # Example usage: Slovak to English
        sk_to_en_translator = Translator(src_lang="sk", tgt_lang="en")
        slovak_sentence = "Ahoj, ako sa máš dnes? Dúfam, že sa ti darí dobre. Toto je slovenčina."
        english_translation = sk_to_en_translator.translate_text(slovak_sentence)
        print(f"Slovak: {slovak_sentence}")
        print(f"English: {english_translation}")

    asyncio.run(run_examples())
