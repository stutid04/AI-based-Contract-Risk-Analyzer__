
import logging
import re

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Cleans and normalizes raw contract text.
    """

    def __init__(self, enable_lemmatization: bool = False):
        self.enable_lemmatization = enable_lemmatization

        if self.enable_lemmatization:
            try:
                import spacy

                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning(
                    "Lemmatization enabled but spaCy model unavailable: %s",
                    e,
                )
                self.enable_lemmatization = False

    def _lemmatize(self, text: str) -> str:
        doc = self.nlp(text)
        return " ".join(token.lemma_ for token in doc)

    def preprocess(self, text: str) -> str:
        """
        Clean and normalize contract text.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """

        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        try:

            # Remove page breaks
            text = text.replace("\f", " ")

            # Remove tabs
            text = text.replace("\t", " ")

            # Normalize line endings
            text = text.replace("\r\n", "\n")
            text = text.replace("\r", "\n")

            # Preserve numbering patterns
            # Keep:
            # 1.
            # 1.1
            # 2.3.4

            text = re.sub(
                r"[^a-zA-Z0-9\s\.\,\;\:\-\(\)\/%\n]",
                " ",
                text,
            )
            # Remove standalone runs of % that are not attached to numbers
            text = re.sub(r"(?<!\d)%+(?!\d)", " ", text)

            # Multiple spaces
            text = re.sub(r"[ ]{2,}", " ", text)

            # Excessive blank lines
            text = re.sub(r"\n\s*\n+", "\n\n", text)

            text = text.lower().strip()

            if self.enable_lemmatization:
                text = self._lemmatize(text)

            return text

        except Exception as exc:
            logger.exception("Preprocessing failed")
            raise RuntimeError(
                f"Text preprocessing failed: {exc}"
            ) from exc
