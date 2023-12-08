from csv import DictReader
from pathlib import Path
import Levenshtein


def score(folder):
    submissions_file = Path(folder) / "submissions.csv"
    ans_file = Path(folder) / "key.csv"
    with open(ans_file.as_posix(), "r") as f:
        answers_reader = DictReader(f)
        for ans_key in answers_reader:
            print(ans_key)
            break

    scores = []

    with open(submissions_file.as_posix(), "r") as f:
        submissions_reader = DictReader(f)
        for submission in submissions_reader:
            count = 0
            for k, v in ans_key.items():
                count += Levenshtein.ratio(submission.get(k, ""), v)
            scores.append((submission["Your Name"], count))

    scores.sort(key=lambda x: x[1], reverse=True)
    print(scores)


if __name__ == "__main__":
    score("movie_game")
