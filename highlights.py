"""
Attention Drop Detector — Highlight Extractor
==============================================
Automatically identifies the most engaging segments in a video.
Useful for creating clips, YouTube Shorts, or highlight reels.

Usage:
  python highlights.py result.json
  python highlights.py result.json --top 5 --min-len 15
"""

import argparse
import json
from dataclasses import dataclass, field

from analyzer import Drop, Result, Window


@dataclass
class Highlight:
    """A contiguous run of high-attention windows."""
    start:       float
    end:         float
    avg_score:   float
    peak_score:  float
    duration:    float
    rank:        int = 0


class HighlightExtractor:
    """
    Merges consecutive engaging/neutral windows into highlight segments,
    ranks them by average score, and returns the top N.
    """

    def extract(
        self,
        windows: list[Window],
        min_score:  float = 0.50,
        min_len:    float = 10.0,
        top_n:      int   = 3,
    ) -> list[Highlight]:
        """
        Args:
            windows:    Analyzed windows from Result.
            min_score:  Minimum attention score to include a window.
            min_len:    Minimum highlight duration in seconds.
            top_n:      How many highlights to return.

        Returns:
            List of Highlight objects, ranked best-first.
        """
        highlights = []
        run_start  = None
        run_wins   = []

        for w in windows:
            if w.score >= min_score:
                if run_start is None:
                    run_start = w.start
                run_wins.append(w)
            else:
                if run_wins:
                    self._flush(run_start, run_wins, min_len, highlights)
                run_start = None
                run_wins  = []

        # Flush final run
        if run_wins:
            self._flush(run_start, run_wins, min_len, highlights)

        # Rank by average score
        highlights.sort(key=lambda h: h.avg_score, reverse=True)
        top = highlights[:top_n]
        for i, h in enumerate(top, 1):
            h.rank = i

        return top

    @staticmethod
    def _flush(
        start: float,
        wins:  list[Window],
        min_len: float,
        out: list[Highlight],
    ) -> None:
        end      = wins[-1].end
        duration = end - start
        if duration < min_len:
            return
        scores = [w.score for w in wins]
        out.append(Highlight(
            start      = start,
            end        = end,
            avg_score  = round(sum(scores) / len(scores), 4),
            peak_score = round(max(scores), 4),
            duration   = round(duration, 2),
        ))


def fmt_time(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}:{sec:02d}"


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract highlight segments from analysis")
    parser.add_argument("result_json", help="Path to result.json from analyzer.py")
    parser.add_argument("--top",     "-n", type=int,   default=3,    help="Number of highlights")
    parser.add_argument("--min-score",     type=float, default=0.50, help="Min attention score")
    parser.add_argument("--min-len",       type=float, default=10.0, help="Min highlight length (seconds)")
    args = parser.parse_args()

    with open(args.result_json) as f:
        data = json.load(f)

    windows = [Window(**w) for w in data["windows"]]
    extractor = HighlightExtractor()
    highlights = extractor.extract(
        windows,
        min_score = args.min_score,
        min_len   = args.min_len,
        top_n     = args.top,
    )

    print(f"\n── Top {args.top} Highlights " + "─" * 40)
    if not highlights:
        print("  No highlights found — try lowering --min-score or --min-len")
    for h in highlights:
        print(f"\n  #{h.rank}  {fmt_time(h.start)} – {fmt_time(h.end)}  ({h.duration:.0f}s)")
        print(f"      avg score : {h.avg_score:.0%}")
        print(f"      peak score: {h.peak_score:.0%}")
        print(f"      → use as clip or Short")
