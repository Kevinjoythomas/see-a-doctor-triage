"""Rule-based emergency red-flag detector (gate variant G-rule).

The lexicon was written from general emergency-care guidance (UK NHS "call 999" and
"go to A&E" lists, US "when to call 911" patient guidance, and standard ED red-flag
teaching: FAST stroke signs, ACS, anaphylaxis, sepsis/meningitis, DKA, ectopic
pregnancy, pre-eclampsia, cauda equina, testicular torsion, overdose/self-harm,
paediatric danger signs). It was NOT derived from any evaluation case. It is tuned only
on the corpus-adjacent development set.

detect(text) -> (fired: bool, hits: list[str])
"""
import re

# each entry: (label, pattern). Patterns are case-insensitive and matched per sentence.
RED_FLAGS = [
    # cardiovascular
    ("chest_pain_radiating", r"chest (pain|pressure|tightness|heaviness)[^.]{0,80}(arm|jaw|neck|back|shoulder|sweat|breath|nause)"),
    ("chest_pain_exertion", r"(crushing|squeezing|heavy|severe) (chest|central) (pain|pressure)"),
    ("chest_pain_plain", r"\bchest (pain|pressure|tightness)\b"),
    ("tearing_pain", r"tearing (pain|sensation)|ripping pain"),
    ("palpitations_faint", r"(heart (racing|pounding)|palpitations)[^.]{0,60}(faint|pass(ed)? out|dizz|chest)"),
    ("bp_crisis", r"\b(1[89]\d|2\d\d)\s*/\s*(1[2-9]\d|\d{3})\b"),
    # neurological
    ("stroke_face_arm_speech", r"(face (is )?(droop|drooping|numb)|droop(ing|y) (face|mouth|side)|slurred speech|can'?t (speak|talk) properly|arm (is )?(weak|numb|won'?t move)|(one|left|right) side of (my|his|her|the) (body|face) (is )?(weak|numb))"),
    ("sudden_vision_loss", r"(sudden(ly)?|suddenly) [^.]{0,30}(lost|loss of|can'?t see|blind|double vision)|lost (the )?vision"),
    ("thunderclap_headache", r"(worst headache|thunderclap|sudden(,)? severe headache|headache[^.]{0,40}(worst|sudden|explod))"),
    ("seizure", r"\b(seizure|seizing|convuls|fit(ting)?\b[^.]{0,20}(won'?t stop|first))"),
    ("unconscious", r"(unconscious|unresponsive|passed out|pass(es|ing)? out|fainted|collapsed|won'?t wake|can'?t wake|hard to wake)"),
    ("confusion_acute", r"(sudden(ly)?|new|acute(ly)?)[^.]{0,20}confus|very confused|not making sense"),
    ("cauda_equina", r"(numb(ness)?[^.]{0,30}(groin|saddle|between (my|the) legs|buttock))|(can'?t|unable to) (control|hold)[^.]{0,20}(bladder|bowel|urine|pee)|lost control of (my )?(bladder|bowel)"),
    ("neck_stiff_fever", r"(stiff neck|neck (is )?stiff)[^.]{0,80}(fever|rash|light|headache)|(fever|rash)[^.]{0,80}stiff neck"),
    ("non_blanching_rash", r"rash[^.]{0,40}(doesn'?t|does not|won'?t) fade|glass test|purple (spots|rash)|purpura|petechia"),
    # respiratory / airway / allergy
    ("cant_breathe", r"(can'?t|cannot|unable to|struggling to|hard to|difficulty) breath|short(ness)? of breath[^.]{0,40}(rest|sudden|severe|worse|lying)|gasping|choking"),
    ("blue_lips", r"(blue|grey|gray) (lips|face|skin|tongue)|lips (are |turning )?blue|cyanos"),
    ("anaphylaxis", r"(throat|tongue|lips?|face) (is |are |feels? )?(swell|swollen|closing|tight)|swelling of (the )?(throat|tongue|lips|face)|anaphyla|epi-?pen"),
    ("coughing_blood", r"cough(ing|ed)? (up )?blood|haemoptysis|hemoptysis"),
    # GI / bleeding / surgical
    ("vomit_blood", r"vomit(ing|ed)? (up )?blood|blood in (my )?vomit|coffee[- ]ground|haematemesis|hematemesis"),
    ("melena", r"(black|tarry|tar-like) (stool|poo|poop)|melaena|melena|passing (a lot of )?blood"),
    ("bleeding_wont_stop", r"(bleeding|blood)[^.]{0,30}(won'?t|doesn'?t|does not|will not) stop|heavy bleeding|soak(ed|ing) (through|a pad)|spurting"),
    ("severe_abdo", r"(severe|worst|unbearable|excruciating|sudden) [^.]{0,20}(abdominal|stomach|belly|tummy) pain|(abdomen|belly|stomach) (is )?(rigid|hard as a board)"),
    ("testicular_torsion", r"(sudden|severe)[^.]{0,30}(testic|testis|testicle|scrot)|(testic|testicle|scrot)[^.]{0,30}(sudden|severe|swollen)"),
    # obstetric
    ("pregnant_bleeding_pain", r"pregnan[^.]{0,80}(bleeding|severe pain|cramp|spotting|fluid|leak)|(bleeding|severe pain)[^.]{0,80}pregnan"),
    ("preeclampsia", r"pregnan[^.]{0,120}(severe headache|blurr|vision|flashing|swelling of (my )?(face|hands))"),
    ("reduced_fetal_movement", r"(baby|fetus|foetus)[^.]{0,30}(not moving|moving less|stopped moving|reduced movement)"),
    # metabolic / infection
    ("dka", r"fruity|ketone|blood sugar[^.]{0,40}(over|above|high|\b[3-6]\d\d\b)[^.]{0,80}(vomit|breath|confus)|(\b[4-6]\d\d\b)[^.]{0,20}(mg|blood sugar|glucose)"),
    ("hypoglycaemia_severe", r"(blood sugar|glucose)[^.]{0,30}(\b[2-4]\d\b|very low)[^.]{0,60}(confus|shak|sweat|unconscious)"),
    ("sepsis", r"(fever|temperature|chills|shiver)[^.]{0,80}(confus|mottled|cold hands|not passing urine|fast breathing|very drowsy)|sepsis|septic"),
    # paediatric danger signs
    ("infant_fever", r"(\b[0-9]|[0-9]{1,2}[- ])(week|day|month)[- ]?old[^.]{0,80}(fever|temperature|38|39|100\.[4-9]|10[1-4])|(newborn|neonate)[^.]{0,60}(fever|temperature)"),
    ("child_floppy", r"(baby|child|toddler|infant|son|daughter)[^.]{0,60}(floppy|limp|lethargic|won'?t wake|not waking|not feeding|refus(ing|es) (to )?feed|no wet nappies|no wet diapers)"),
    # toxicology / self-harm
    ("overdose", r"overdos|took (too many|a lot of|a bunch of|the whole|\d+ )[^.]{0,30}(pills|tablets|paracetamol|acetaminophen|tylenol)|swallowed (bleach|battery|batteries|poison|cleaner|antifreeze)"),
    ("poisoning", r"poison|carbon monoxide|\bco alarm|antifreeze|bleach|chemical (in|on) (my )?(eye|skin)"),
    ("suicidal", r"(kill|hurt|harm) (my ?self|myself)|suicid|want to die|end (my|it all)|don'?t want to (live|be alive)|no reason to live"),
    # trauma / environment
    ("head_injury_red", r"(hit|bang|banged|injur)[^.]{0,30}head[^.]{0,80}(vomit|confus|drows|blood thinner|warfarin|unconscious|pass)"),
    ("major_trauma", r"(car (crash|accident)|fell from|fall from|hit by a car|stab|gunshot|shot\b|deep (cut|wound)|bone (is )?(sticking|poking) out)"),
    ("burn_severe", r"(severe|large|deep) burn|burn[^.]{0,40}(face|airway|large|blister(s|ed)? all over)|electr(ic|ocut)[^.]{0,30}shock"),
    ("heat_cold", r"heat ?stroke|stopped sweating|hypotherm"),
    ("dvt_pe", r"(calf|leg)[^.]{0,40}(swollen|swelling)[^.]{0,80}(breath|chest)|(breath|chest)[^.]{0,80}(calf|leg)[^.]{0,30}(swollen|swelling)"),
]

NEGATORS = re.compile(r"\b(no|not|never|denies|deny|without|negative for|free of)\b[^.]{0,25}$", re.I)
_COMPILED = [(lab, re.compile(p, re.I)) for lab, p in RED_FLAGS]
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")

# labels that fire the gate on their own; the rest are counted as "weak" signals
STRONG = {lab for lab, _ in RED_FLAGS} - {"chest_pain_plain", "poisoning", "major_trauma"}


def detect(text, min_weak=2):
    """Return (fired, hits). Fires on any STRONG hit, or on >= min_weak weak hits."""
    hits = []
    for sent in _SENT.split(text or ""):
        for lab, rx in _COMPILED:
            m = rx.search(sent)
            if not m:
                continue
            if NEGATORS.search(sent[:m.start()]):
                continue
            hits.append(lab)
    hits = sorted(set(hits))
    strong = [h for h in hits if h in STRONG]
    weak = [h for h in hits if h not in STRONG]
    return bool(strong) or len(weak) >= min_weak, hits


def score(text):
    """Continuous score for AUROC: strong hits count 1.0, weak hits 0.5."""
    _, hits = detect(text)
    return sum(1.0 if h in STRONG else 0.5 for h in hits)
