"""
One place for the letter-slot repair rule, imported by the benchmarks.

It is deliberately not in app/: the rule is NOT part of the service. It was
measured on the labelled dataset and rejected, because guessing which letter a
0 stands for (O, Q or D) is wrong about a quarter of the time, which would send
OCRM to a different car. It stays here as a diagnostic: it tells us whether a
failed reading belongs to that confusion family or to something else.

Two benchmarks use it, so it lives in its own module. Copying it into both was
the mistake that made bench/ocr_dataset_eval.py report stale numbers after the
pipeline changed.
"""

PLATE_LEN = 8
LETTER_SLOTS = (3, 4, 5)
DIGIT_TO_LETTER = {"0": ("O", "Q", "D"), "1": ("I",), "2": ("Z",), "4": ("A",),
                   "5": ("S",), "6": ("G",), "7": ("T",), "8": ("B",)}


def repair_letter_slots(text):
    """(plate, certainty) for a reading whose letter slots hold digits.

    Scans every 8-character window the way extract_plate does and answers only
    when exactly one candidate comes out. certainty is "однозначно" when every
    repaired digit maps to a single letter, "догадка" when at least one digit
    is ambiguous. ("", "") means nothing fits.
    """
    found = {}
    for start in range(len(text) - PLATE_LEN + 1):
        window = text[start:start + PLATE_LEN]
        if not (window[:3].isdigit() and window[6:].isdigit()):
            continue
        if all(window[i].isalpha() for i in LETTER_SLOTS):
            continue          # already a plate; extract_plate handles this one
        chars, guessed, fits = list(window), False, True
        for i in LETTER_SLOTS:
            if window[i].isalpha():
                continue
            options = DIGIT_TO_LETTER.get(window[i])
            if not options:
                fits = False
                break
            chars[i] = options[0]
            guessed = guessed or len(options) > 1
        if fits:
            found["".join(chars)] = "догадка" if guessed else "однозначно"
    if len(found) != 1:
        return "", ""
    return next(iter(found.items()))


def repairs_to(text, wanted):
    """Does this reading repair to one of `wanted`, ignoring which letter we'd guess?

    Used to attribute a failed reading to a known plate: every letter slot that
    came back as a digit is allowed to be any of its candidate letters, so
    "70700009" is recognised as a damaged reading of "707QQQ09" without the
    rule having to guess Q over O. Returns the matching plate or "".
    """
    for plate in wanted:
        if len(plate) != PLATE_LEN:
            continue
        for start in range(len(text) - PLATE_LEN + 1):
            window = text[start:start + PLATE_LEN]
            if window == plate:
                continue                      # a clean read, not a repair
            if window[:3] != plate[:3] or window[6:] != plate[6:]:
                continue
            if all(window[i] == plate[i]
                   or plate[i] in DIGIT_TO_LETTER.get(window[i], ())
                   for i in LETTER_SLOTS):
                return plate
    return ""
