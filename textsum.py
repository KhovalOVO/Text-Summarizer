from lxml import etree
from nltk import download, sent_tokenize, word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
download('stopwords')

class Summerizer:
    def __init__(self):
        self.stop_words = stopwords.words('english') + list(punctuation)
        self.lemmatizer = WordNetLemmatizer()
        self.model = TfidfVectorizer(tokenizer=word_tokenize)

    def get_root(self, path: str):
        return etree.parse(path).getroot()[0]

    def get_text(self, data):
        return data[0].text, data[1].text

    def get_sents(self, data):
        return sent_tokenize(data)

    def get_words(self, sentence):
        return word_tokenize(sentence.lower())

    def remove_stop(self, tokens):
        return [w for w in tokens if w not in self.stop_words]

    def get_lemma(self, token):
        return self.lemmatizer.lemmatize(token, pos='n')

    def get_tfidf(self, data, header_lemmas, xtra_weight=3):
        tfidf_matrix = self.model.fit_transform(data).toarray()
        words = self.model.get_feature_names_out()

        for i, row in enumerate(tfidf_matrix):
            for j, score in enumerate(row):
                if words[j] in header_lemmas:
                    tfidf_matrix[i][j] *= xtra_weight

        return tfidf_matrix


if __name__ == "__main__":
    summ = Summerizer()
    root = summ.get_root("news.xml")
    for news in root:
        header, text = summ.get_text(news)
        sents = summ.get_sents(text)
        num_of_sents = round(np.sqrt(len(sents)))
        lemmas = []
        token_header = summ.get_words(header)
        filtered_header = summ.remove_stop(token_header)
        lemmas_header = [summ.get_lemma(word) for word in filtered_header]
        for sentence in sents:
            w_tokens = summ.get_words(sentence)
            filtered_tokens = summ.remove_stop(w_tokens)
            lemmas.append(" ".join([summ.get_lemma(lemma) for lemma in filtered_tokens]))
        scored_sent = summ.get_tfidf(lemmas, lemmas_header)
        scores = {}
        for i in range(len(lemmas)):
            scores[sents[i]] = np.mean([calc for calc in scored_sent[i] if calc > 0])
        sorted_scores = dict(sorted(scores.items(), key=lambda x: -x[1])[:num_of_sents])
        ordered_scores = [key for key in sents if key in sorted_scores.keys()]
        print("HEADER:", header)
        print("TEXT:", "\n".join(ordered_scores))
        print()


