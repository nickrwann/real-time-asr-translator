#!/usr/bin/env python3
"""
translate_text.py

A simple REPL CLI tool that prompts the user to enter text in English or Spanish,
then outputs both the English and Spanish versions for each input.

Usage:
    pip install argostranslate langdetect
    python translate_text.py

Type 'exit' or press Ctrl+C to quit.
"""
import sys
import argostranslate.package
import argostranslate.translate
from langdetect import detect, DetectorFactory

# Make language detection deterministic
DetectorFactory.seed = 0

def ensure_argos_models():
    """Install English<->Spanish models if not already present."""
    # Refresh available package index
    argostranslate.package.update_package_index()

    # Determine installed language pairs
    installed_pairs = {
        (pkg.from_code, pkg.to_code)
        for pkg in argostranslate.package.get_installed_packages()
    }

    # Install en->es and es->en if missing
    for pkg in argostranslate.package.get_available_packages():
        pair = (pkg.from_code, pkg.to_code)
        if pair in (("en", "es"), ("es", "en")) and pair not in installed_pairs:
            print(f"Installing model {pair[0]}->{pair[1]}…", file=sys.stderr)
            path = pkg.download()
            argostranslate.package.install_from_path(path)
            print("Installation complete.", file=sys.stderr)


def load_translators():
    """Load translator objects for English and Spanish."""
    langs = argostranslate.translate.get_installed_languages()
    lang_en = next(l for l in langs if l.code == "en")
    lang_es = next(l for l in langs if l.code == "es")
    return (
        lang_en.get_translation(lang_es),  # English -> Spanish
        lang_es.get_translation(lang_en),  # Spanish -> English
    )


def translate_text(text, en_to_es, es_to_en):
    """Detect language of input text and print both English and Spanish."""
    try:
        lang = detect(text)
    except Exception:
        lang = "en"

    if lang.startswith("es"):
        spanish = text
        english = es_to_en.translate(text)
    else:
        english = text
        spanish = en_to_es.translate(text)

    print(f"English: {english}")
    print(f"Spanish: {spanish}")


def main():
    ensure_argos_models()
    en_to_es, es_to_en = load_translators()

    print("Enter text in English or Spanish (type 'exit' to quit)")
    while True:
        try:
            text = input("> ").strip()
            if not text:
                continue
            if text.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            translate_text(text, en_to_es, es_to_en)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
