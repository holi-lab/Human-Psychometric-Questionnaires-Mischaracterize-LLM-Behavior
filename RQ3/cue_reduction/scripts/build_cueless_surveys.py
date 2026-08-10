#!/usr/bin/env python3
"""
Build cue-removed versions of PVQ-40 and BFI-44.

Rewrites are authored by hand (below). This script merely merges them with the
original survey files so that meta_data (construct, scoring, higher-order tags)
is copied VERBATIM from the originals, and emits:

    surveys_cueless/PVQ.json
    surveys_cueless/BFI44.json
    surveys_cueless/mapping.json

Design rules for the rewrites:
  - Remove overt lexical construct signals (trait adjectives, value vocabulary,
    words appearing in the construct definitions used by RQ3).
  - Preserve the underlying disposition and the item polarity
    (reverse-scored items remain reverse-keyed).
  - Keep the sentence frame: PVQ third-person portraits (2 sentences),
    BFI predicates completing "I see myself as someone who ...".
  - Describe concrete behavior/situations (VP-scenario style) instead of
    naming the disposition.

Usage:
    python RQ3/cue_reduction/scripts/build_cueless_surveys.py
"""

import json
from pathlib import Path

CUE_DIR = Path(__file__).resolve().parent.parent
ROOT = CUE_DIR.parent.parent  # Official/ repo root
OUT_DIR = CUE_DIR / "surveys_cueless"

# ─────────────────────────────────────────────────────────────────────────────
# PVQ-40 rewrites: id -> (rewritten_text, removed_cues, note)
# ─────────────────────────────────────────────────────────────────────────────
PVQ_REWRITES = {
    "1": (
        "When they take on a task, they would rather work out their own way of "
        "doing it than follow how it was done before. They often sketch out "
        "possibilities that nobody around them has suggested.",
        ["thinking up new ideas", "creative", "original"],
        "Self-Direction (creativity) expressed as concrete work behavior.",
    ),
    "2": (
        "It matters to them that their bank balance keeps growing year after "
        "year. They want the kind of house, car, and watch that most people "
        "could never afford.",
        ["rich", "a lot of money", "expensive things"],
        "Power (wealth) via concrete possessions; avoids 'rich'/'money'.",
    ),
    "3": (
        "It bothers them when a child born in one neighborhood gets chances "
        "that a child born a few streets away will never see. They think the "
        "same doors should open for everyone, everywhere.",
        ["treated equally", "equal opportunities"],
        "Universalism (equality) as a concrete contrast case.",
    ),
    "4": (
        "When they finish a difficult piece of work, they make sure the people "
        "who matter get to see it. Hearing others talk about what they have "
        "pulled off makes their day.",
        ["show their abilities", "admire"],
        "Achievement (recognition-seeking) behaviorally.",
    ),
    "5": (
        "They double-check the locks before bed and prefer streets where "
        "nothing much ever happens. They stay away from shortcuts through "
        "areas they don't know.",
        ["secure surroundings", "endanger", "safety"],
        "Security (personal) as routines; no safety vocabulary.",
    ),
    "6": (
        "Their weekends rarely look the same twice — one month a pottery "
        "class, the next a road trip. If a menu lists a dish they can't "
        "pronounce, that's the one they order.",
        ["lots of different things", "new things to try"],
        "Stimulation (variety) via concrete episodes.",
    ),
    "7": (
        "They stop at the red light at an empty intersection at 3 a.m. and "
        "wait for it to change. When a supervisor gives an instruction, they "
        "carry it out without arguing.",
        ["do what they're told", "follow rules"],
        "Conformity (rule-following/obedience) as concrete behavior.",
    ),
    "8": (
        "At a gathering, they sit down next to the person whose views clash "
        "with their own and ask questions instead of walking away. They want "
        "to hear the full story before making up their mind about anyone.",
        ["people who are different from them", "understand them"],
        "Universalism (tolerance) behaviorally.",
    ),
    "9": (
        "Once their needs are met, they stop chasing a bigger house or a "
        "better job title. In their eyes, someone who keeps wanting more has "
        "missed the point.",
        ["not to ask for more", "satisfied with what they have"],
        "Tradition (moderation/acceptance of one's portion).",
    ),
    "10": (
        "If there is a party on Friday and a long lunch on Sunday, they will "
        "be at both. Given a free afternoon, they spend it on whatever feels "
        "best in the moment.",
        ["have fun", "pleasure"],
        "Hedonism without 'fun'/'pleasure'.",
    ),
    "11": (
        "They would turn down a well-paid job if someone else got to set "
        "their schedule. Nobody fills in their calendar but them.",
        ["make their own decisions", "free to plan", "choose for themselves"],
        "Self-Direction (autonomy) via a concrete tradeoff.",
    ),
    "12": (
        "When a friend is moving apartments or stranded at the airport, they "
        "are the first phone call. They keep an eye on how the people in "
        "their circle are holding up.",
        ["help the people around them", "care for their well-being"],
        "Benevolence (helping close others) behaviorally.",
    ),
    "13": (
        "They want their name near the top when the results are posted. It "
        "matters to them that others raise their eyebrows at what they have "
        "done.",
        ["successful", "impress"],
        "Achievement (success/impressing) without the keywords.",
    ),
    "14": (
        "They follow news about border incidents closely and want more "
        "spending on the army and the police. Groups that could stir up "
        "trouble from within make them uneasy.",
        ["country be safe", "threats", "the state must be on watch"],
        "Security (national) via concrete policy preferences.",
    ),
    "15": (
        "They sign up for the cliff jump while the others watch from the "
        "shore. A plan with an uncertain ending draws them more than one "
        "with a guaranteed outcome.",
        ["take risks", "adventures"],
        "Stimulation (risk) via a concrete scene.",
    ),
    "16": (
        "Before doing anything, they ask themselves what the people around "
        "them would make of it. They would rather pass up something they "
        "wanted than give the neighbors a reason to talk.",
        ["behave properly", "wrong"],
        "Conformity (propriety) as anticipatory self-restraint.",
    ),
    "17": (
        "In any group project, they end up assigning the tasks and setting "
        "the deadlines. Things go smoother, they feel, when everyone else "
        "simply follows their plan.",
        ["be in charge", "tell others what to do"],
        "Power (dominance) behaviorally.",
    ),
    "18": (
        "If someone they have known for years calls at midnight, they get in "
        "the car. The handful of people they grew up with can count on them "
        "for anything.",
        ["loyal", "devote themselves", "people close to them"],
        "Benevolence (loyalty) via a concrete episode.",
    ),
    "19": (
        "They sort their recycling, cycle instead of driving when they can, "
        "and give money to groups that replant forests. A riverbank covered "
        "in plastic bags genuinely upsets them.",
        ["care for nature", "environment"],
        "Universalism (nature) via concrete pro-environmental behaviors.",
    ),
    "20": (
        "They set aside time every week to attend services, and they keep "
        "the fasts and observances they were raised with. Before big "
        "decisions, they pray.",
        ["religious belief", "religion"],
        "Tradition (religiosity); concrete practice words (pray/services) "
        "are unavoidable to preserve meaning.",
    ),
    "21": (
        "Every jar in their kitchen has a label, and every shirt in the "
        "closet faces the same way. A sink of dishes left overnight bothers "
        "them more than they would like to admit.",
        ["organized", "clean", "mess"],
        "Security (order/cleanliness) via concrete household detail.",
    ),
    "22": (
        "An offhand question about how bridges stay up can cost them an "
        "evening of reading. They ask 'why' more often than most adults "
        "they know.",
        ["interested in things", "curious", "understand all sorts of things"],
        "Self-Direction (curiosity) behaviorally.",
    ),
    "23": (
        "News of a ceasefire anywhere on the globe makes their week. They "
        "give to organizations that bring together groups who have been "
        "fighting for generations.",
        ["live in harmony", "promoting peace"],
        "Universalism (world peace) via concrete behavior.",
    ),
    "24": (
        "They set targets that make other people whistle, and then chase "
        "them down. When a hard problem has beaten everyone else, they "
        "volunteer — partly so everyone can watch them solve it.",
        ["ambitious", "show how capable they are"],
        "Achievement (ambition) behaviorally.",
    ),
    "25": (
        "They cook the dishes their grandmother cooked, the same way, on the "
        "same dates every year. If something has been done a certain way for "
        "generations, they see no reason to change it.",
        ["traditional ways", "customs"],
        "Tradition without 'tradition'/'customs'.",
    ),
    "26": (
        "They book the massage, order the dessert, and run the long bath "
        "without a second thought. Treating themselves well is a fixed line "
        "in their budget, not an afterthought.",
        ["enjoying life's pleasures", "spoil themselves"],
        "Hedonism (self-indulgence) via concrete treats.",
    ),
    "27": (
        "When someone in their circle is short on rent or buried in work, "
        "they quietly step in. People around them have learned that asking "
        "once is enough.",
        ["respond to the needs of others", "support those they know"],
        "Benevolence (responsiveness) behaviorally.",
    ),
    "28": (
        "They still answer their father with 'sir' and would never "
        "contradict a grandparent at the table. When someone senior asks "
        "them to do something, they do it without argument.",
        ["show respect", "obedient"],
        "Conformity (deference to elders) via concrete behavior.",
    ),
    "29": (
        "They speak up when a stranger is being shortchanged, not only when "
        "it happens to a friend. They give time and money to shelters and "
        "legal-aid clinics for people who cannot push back on their own.",
        ["treated justly", "protect the weak"],
        "Universalism (justice) via concrete actions.",
    ),
    "30": (
        "They would rather open an unmarked envelope than be told what is "
        "inside. A year in which nothing unexpected happened would feel like "
        "a year lost.",
        ["surprises", "exciting life"],
        "Stimulation (novelty-seeking) without the keywords.",
    ),
    "31": (
        "They book every recommended check-up, keep sanitizer in the car, "
        "and go to bed early when something is going around. Their medicine "
        "cabinet is stocked before anyone sneezes.",
        ["avoid getting sick", "staying healthy"],
        "Security (health) via concrete precautions.",
    ),
    "32": (
        "They keep track of where they stand against the people they started "
        "out with — pay, title, results — and second place stings. "
        "Each year they aim to be a rung above last year.",
        ["getting ahead in life", "do better than others"],
        "Achievement (social comparison) behaviorally.",
    ),
    "33": (
        "After a friend let them down badly, they still showed up at his "
        "birthday a month later. They would rather remember the good years "
        "than replay one bad evening.",
        ["forgiving", "hold a grudge"],
        "Benevolence (forgiveness) via a concrete episode.",
    ),
    "34": (
        "They fixed the leaking pipe with a video tutorial rather than call "
        "their brother-in-law. Handing their affairs over to someone else "
        "makes them uncomfortable.",
        ["independent", "rely on themselves"],
        "Self-Direction (self-reliance) via a concrete episode.",
    ),
    "35": (
        "They vote for candidates who promise continuity rather than "
        "upheaval. Strikes, sudden reforms, and street protests make them "
        "uneasy about where the country is heading.",
        ["stable government", "social order be protected"],
        "Security (societal stability) via concrete political preferences.",
    ),
    "36": (
        "They apologize when someone else bumps into them, and keep their "
        "music low even at midday. They would take the longer hallway rather "
        "than walk between two people talking.",
        ["polite", "disturb or irritate others"],
        "Conformity (politeness) behaviorally.",
    ),
    "37": (
        "Their calendar fills with dinners, festivals, and weekends away "
        "before anything else gets a slot. If an evening ends with cheeks "
        "sore from laughing, it counted.",
        ["enjoy life", "having a good time"],
        "Hedonism (enjoyment) via concrete episodes.",
    ),
    "38": (
        "When their project wins, they nudge a teammate forward to collect "
        "the prize. They pick the seat at the back of the room and change "
        "the subject when the conversation turns to them.",
        ["humble", "modest", "draw attention to themselves"],
        "Tradition (humility) behaviorally.",
    ),
    "39": (
        "When the group cannot settle on a restaurant, they name one, and "
        "everyone goes. Committees they join tend to end up with them "
        "holding the gavel — which suits them fine.",
        ["makes the decisions", "leader"],
        "Power (leadership/decision control) behaviorally.",
    ),
    "40": (
        "They would rather a footpath bend around an old oak than the oak be "
        "cut for a straighter path. When they camp, they leave the clearing "
        "exactly as they found it.",
        ["adapt to nature", "fit into it", "change nature"],
        "Universalism (unity with nature) via concrete choices.",
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# BFI-44 rewrites: id -> (rewritten_predicate, removed_cues, note)
# Predicates complete the stem "I see myself as someone who ...".
# ─────────────────────────────────────────────────────────────────────────────
BFI_REWRITES = {
    "1": (
        "keeps a conversation going long after others have run out of things "
        "to say.",
        ["talkative"], "Extraversion (+).",
    ),
    "2": (
        "usually notices what a colleague got wrong before noticing what "
        "they got right.",
        ["find fault with others"], "Agreeableness (reverse) preserved.",
    ),
    "3": (
        "reads a document down to the last page before signing off on it.",
        ["thorough"], "Conscientiousness (+).",
    ),
    "4": (
        "has stretches where getting out of bed takes real effort and little "
        "seems worth doing.",
        ["depressed", "blue"], "Neuroticism (+), depressive mood behaviorally.",
    ),
    "5": (
        "suggests approaches in meetings that nobody else at the table had "
        "thought of.",
        ["original", "new ideas"], "Openness (+).",
    ),
    "6": (
        "lets others do most of the talking at gatherings.",
        ["reserved"], "Extraversion (reverse) preserved.",
    ),
    "7": (
        "gives up a free weekend to move a friend's furniture without being "
        "asked twice.",
        ["helpful", "unselfish"], "Agreeableness (+).",
    ),
    "8": (
        "sometimes sends the email before noticing the attachment is missing.",
        ["careless"], "Conscientiousness (reverse) preserved.",
    ),
    "9": (
        "sleeps soundly the night before a big exam or presentation.",
        ["relaxed", "handles stress well"], "Neuroticism (reverse) preserved.",
    ),
    "10": (
        "will happily lose an afternoon reading about topics that have "
        "nothing to do with their work.",
        ["curious"], "Openness (+).",
    ),
    "11": (
        "is still going strong when the rest of the group is ready to head "
        "home.",
        ["full of energy"], "Extraversion (+).",
    ),
    "12": (
        "turns small disagreements over dinner into raised voices.",
        ["starts quarrels"], "Agreeableness (reverse) preserved.",
    ),
    "13": (
        "delivers what was promised on the day it was promised.",
        ["reliable worker"], "Conscientiousness (+).",
    ),
    "14": (
        "sits with a clenched jaw and tight shoulders when things are up in "
        "the air.",
        ["tense"], "Neuroticism (+).",
    ),
    "15": (
        "keeps turning a question over long after everyone else has settled "
        "for the easy answer.",
        ["ingenious", "deep thinker"], "Openness (+).",
    ),
    "16": (
        "can get a whole team eager to start on a plan they were lukewarm "
        "about an hour earlier.",
        ["enthusiasm"], "Extraversion (+).",
    ),
    "17": (
        "meets a friend for coffee a week after that friend let them down "
        "badly.",
        ["forgiving nature"], "Agreeableness (+).",
    ),
    "18": (
        "hunts for keys, chargers, and paperwork nearly every morning.",
        ["disorganized"], "Conscientiousness (reverse) preserved.",
    ),
    "19": (
        "lies awake replaying tomorrow's appointments and what could go "
        "wrong with each.",
        ["worries"], "Neuroticism (+).",
    ),
    "20": (
        "can spend a commute inventing whole life stories for the strangers "
        "across the aisle.",
        ["active imagination"], "Openness (+).",
    ),
    "21": (
        "can share a long car ride without feeling any need to fill the "
        "silence.",
        ["quiet"], "Extraversion (reverse) preserved.",
    ),
    "22": (
        "takes people's stories at face value unless given a clear reason "
        "not to.",
        ["trusting"], "Agreeableness (+).",
    ),
    "23": (
        "leaves the laundry until there is nothing left to wear.",
        ["lazy"], "Conscientiousness (reverse) preserved.",
    ),
    "24": (
        "takes a canceled flight or a harsh comment without letting it ruin "
        "the day.",
        ["emotionally stable", "not easily upset"],
        "Neuroticism (reverse) preserved.",
    ),
    "25": (
        "rigs a working fix out of spare parts when the proper tool is "
        "missing.",
        ["inventive"], "Openness (+).",
    ),
    "26": (
        "says plainly at the table what they want and where they stand.",
        ["assertive personality"], "Extraversion (+).",
    ),
    "27": (
        "answers a coworker's warm greeting with a short nod and keeps "
        "walking.",
        ["cold and aloof"], "Agreeableness (reverse) preserved.",
    ),
    "28": (
        "stays at the desk until the last box on the list is ticked, even "
        "past midnight.",
        ["perseveres until the task is finished"], "Conscientiousness (+).",
    ),
    "29": (
        "can go from cheerful at breakfast to snappish by lunch without an "
        "obvious reason.",
        ["moody"], "Neuroticism (+).",
    ),
    "30": (
        "will cross town on a weeknight to catch an exhibition or hear a "
        "string quartet.",
        ["values artistic, aesthetic experiences"],
        "Openness (+); art-domain content kept, trait adjective removed.",
    ),
    "31": (
        "rehearses a sentence twice before saying it to a stranger at a "
        "party.",
        ["shy", "inhibited"], "Extraversion (reverse) preserved.",
    ),
    "32": (
        "remembers the doorman's name and asks how his daughter's exams "
        "went.",
        ["considerate and kind"], "Agreeableness (+).",
    ),
    "33": (
        "clears in one afternoon what takes others most of a week.",
        ["efficiently"], "Conscientiousness (+).",
    ),
    "34": (
        "was the one giving directions when the fire alarm went off.",
        ["remains calm", "tense situations"],
        "Neuroticism (reverse) preserved.",
    ),
    "35": (
        "is happiest when today's tasks look exactly like yesterday's.",
        ["routine"], "Openness (reverse) preserved.",
    ),
    "36": (
        "leaves a party with three new phone numbers.",
        ["outgoing", "sociable"], "Extraversion (+).",
    ),
    "37": (
        "sometimes cuts the waiter off mid-sentence with a sharp remark.",
        ["rude"], "Agreeableness (reverse) preserved.",
    ),
    "38": (
        "books the flights in January and is standing at the gate in June, "
        "itinerary in hand.",
        ["makes plans and follows through"], "Conscientiousness (+).",
    ),
    "39": (
        "feels their heart race when the phone rings from an unknown number.",
        ["nervous"], "Neuroticism (+).",
    ),
    "40": (
        "can chew on a hypothetical question for the length of a dinner.",
        ["reflect", "play with ideas"], "Openness (+).",
    ),
    "41": (
        "walks past the gallery to get to the food court.",
        ["artistic interests"], "Openness (reverse) preserved.",
    ),
    "42": (
        "would rather split the work and share the credit than keep both to "
        "themselves.",
        ["cooperate"], "Agreeableness (+).",
    ),
    "43": (
        "opens the laptop to file a report and surfaces an hour later, "
        "twelve tabs deep, report untouched.",
        ["easily distracted"], "Conscientiousness (reverse) preserved.",
    ),
    "44": (
        "can tell you which translation of a novel to pick and why one "
        "recording of a symphony outshines another.",
        ["sophisticated in art, music, or literature"],
        "Openness (+); domain content kept, trait adjective removed.",
    ),
}


def build(survey_file: str, rewrites: dict, construct_key: str, survey_name: str):
    original = json.loads((ROOT / "surveys" / survey_file).read_text())
    assert set(original.keys()) == set(rewrites.keys()), (
        f"{survey_name}: id mismatch "
        f"{set(original) ^ set(rewrites)}"
    )
    cueless = {}
    mapping = []
    for item_id in sorted(original, key=int):
        text, cues, note = rewrites[item_id]
        meta = original[item_id]["meta_data"]
        cueless[item_id] = {
            "en_neutral": text,
            "meta_data": dict(meta),  # verbatim copy: construct + scoring tags
        }
        mapping.append({
            "id": item_id,
            "survey": survey_name,
            "construct": meta[construct_key],
            "scoring": meta.get("scoring", "normal"),
            "original": original[item_id]["en_neutral"],
            "rewritten": text,
            "removed_cues": cues,
            "note": note,
        })
    return cueless, mapping


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pvq, pvq_map = build("PVQ.json", PVQ_REWRITES, "value", "PVQ")
    bfi, bfi_map = build("BFI44.json", BFI_REWRITES, "trait", "BFI44")

    (OUT_DIR / "PVQ.json").write_text(json.dumps(pvq, indent=2, ensure_ascii=False))
    (OUT_DIR / "BFI44.json").write_text(json.dumps(bfi, indent=2, ensure_ascii=False))
    (OUT_DIR / "mapping.json").write_text(
        json.dumps(pvq_map + bfi_map, indent=2, ensure_ascii=False)
    )
    print(f"Wrote {len(pvq)} PVQ + {len(bfi)} BFI44 cueless items to {OUT_DIR}")

    # sanity checks
    for item_id, entry in bfi.items():
        assert entry["meta_data"]["scoring"] in ("normal", "reverse")
    n_rev = sum(1 for e in bfi.values() if e["meta_data"]["scoring"] == "reverse")
    print(f"BFI44 reverse-keyed items: {n_rev} (expected 16)")


if __name__ == "__main__":
    main()
