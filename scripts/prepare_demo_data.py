"""Generate synthetic demo data for Criteria Bind.

This script generates realistic clinical notes with multiple criteria
for training and testing the criteria-bind pipeline. It creates both
training and test datasets with varied clinical terminology.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

from criteriabind.schemas import Sample


# Clinical note templates with medical terminology
CLINICAL_NOTES = [
    {
        "note": (
            "Patient presents with persistent depressed mood for the past 3 weeks. "
            "Reports significant weight loss (10 lbs) without dieting. "
            "Exhibits psychomotor retardation and difficulty concentrating. "
            "Denies suicidal ideation. Sleep disturbance noted with early morning awakening. "
            "Patient reports anhedonia and fatigue throughout the day."
        ),
        "criteria": [
            "Depressed mood most of the day",
            "Significant weight loss or decrease in appetite",
            "Psychomotor agitation or retardation",
            "Fatigue or loss of energy",
            "Diminished ability to concentrate",
        ],
    },
    {
        "note": (
            "Patient reports recurrent panic attacks with palpitations and sweating. "
            "Episodes occur 2-3 times per week without identifiable triggers. "
            "Patient avoids crowded places due to fear of having another attack. "
            "No history of substance abuse. Physical exam unremarkable. "
            "Heart rate elevated during interview (110 bpm)."
        ),
        "criteria": [
            "Recurrent unexpected panic attacks",
            "Persistent concern about having additional attacks",
            "Significant change in behavior related to attacks",
            "Palpitations or accelerated heart rate",
        ],
    },
    {
        "note": (
            "Patient demonstrates pressured speech and flight of ideas. "
            "Reports sleeping only 3 hours per night for the past week but denies fatigue. "
            "Started multiple new projects and excessive spending noted. "
            "Grandiose beliefs about special abilities. "
            "Family reports increasingly risky behavior and irritability."
        ),
        "criteria": [
            "Decreased need for sleep",
            "More talkative than usual or pressure to keep talking",
            "Flight of ideas or racing thoughts",
            "Increase in goal-directed activity",
            "Excessive involvement in activities with high potential for painful consequences",
        ],
    },
    {
        "note": (
            "Patient reports hearing voices commenting on their actions. "
            "Auditory hallucinations present for 6 months. "
            "Believes neighbors are plotting against them. Disorganized speech noted. "
            "Poor hygiene and self-care. Affect is flat. "
            "Denies current substance use but history of cannabis use."
        ),
        "criteria": [
            "Delusions present",
            "Hallucinations present",
            "Disorganized speech",
            "Grossly disorganized or catatonic behavior",
            "Negative symptoms (diminished emotional expression)",
        ],
    },
    {
        "note": (
            "Patient presents with persistent worry about multiple concerns. "
            "Reports difficulty controlling worry for the past 8 months. "
            "Experiences muscle tension, restlessness, and irritability. "
            "Sleep onset insomnia noted. Concentration difficulties affecting work performance. "
            "No significant medical history."
        ),
        "criteria": [
            "Excessive anxiety and worry",
            "Difficulty controlling worry",
            "Restlessness or feeling on edge",
            "Muscle tension",
            "Sleep disturbance",
        ],
    },
    {
        "note": (
            "Patient reports intrusive thoughts about contamination. "
            "Engages in hand washing rituals 30-40 times daily. "
            "Recognizes behaviors as excessive but unable to stop. "
            "Significant distress and time consumption (3-4 hours daily). "
            "Avoids public restrooms and doorknobs. Skin on hands is raw and cracked."
        ),
        "criteria": [
            "Recurrent and persistent thoughts, urges, or images",
            "Attempts to ignore or suppress thoughts",
            "Repetitive behaviors or mental acts",
            "Behaviors aimed at preventing or reducing anxiety",
            "Obsessions or compulsions are time-consuming",
        ],
    },
    {
        "note": (
            "Veteran presents with nightmares related to combat experience. "
            "Avoids watching war movies or news about military conflicts. "
            "Hypervigilance noted during interview. Exaggerated startle response. "
            "Reports feeling detached from family members. "
            "Difficulty remembering key aspects of traumatic event."
        ),
        "criteria": [
            "Recurrent distressing dreams related to traumatic event",
            "Efforts to avoid activities that arouse recollections",
            "Hypervigilance",
            "Exaggerated startle response",
            "Feeling of detachment or estrangement from others",
        ],
    },
    {
        "note": (
            "Patient reports binge eating episodes twice weekly for past 4 months. "
            "Consumes large amounts of food rapidly with feeling of loss of control. "
            "Marked distress regarding binge eating. "
            "Eating until uncomfortably full reported. "
            "No compensatory behaviors noted. BMI is 32."
        ),
        "criteria": [
            "Recurrent episodes of binge eating",
            "Eating much more rapidly than normal",
            "Eating until feeling uncomfortably full",
            "Marked distress regarding binge eating",
            "Binge eating occurs at least once a week",
        ],
    },
    {
        "note": (
            "Child presents with difficulty sustaining attention in school. "
            "Teacher reports frequent fidgeting and difficulty remaining seated. "
            "Often interrupts others and has trouble waiting turn. "
            "Symptoms present since age 6, now age 9. "
            "Academic performance declining. Forgets daily activities."
        ),
        "criteria": [
            "Often fails to give close attention to details",
            "Often fidgets with hands or feet",
            "Often has difficulty waiting turn",
            "Often interrupts or intrudes on others",
            "Several symptoms present before age 12",
        ],
    },
    {
        "note": (
            "Patient reports extreme fear of social situations. "
            "Avoids public speaking and eating in front of others. "
            "Fear of negative evaluation is pronounced. "
            "Social situations almost always provoke anxiety. "
            "Recognition that fear is excessive. Duration of 10 months."
        ),
        "criteria": [
            "Marked fear or anxiety about social situations",
            "Fear of negative evaluation by others",
            "Social situations almost always provoke fear",
            "Social situations are avoided or endured with intense fear",
            "Fear is out of proportion to actual threat",
        ],
    },
    {
        "note": (
            "Adolescent patient displays pattern of angry outbursts. "
            "Episodes occur 3-4 times per week for past year. "
            "Irritable mood persists between outbursts. "
            "Arguments with parents and teachers are frequent. "
            "Deliberately annoys siblings. Blames others for mistakes."
        ),
        "criteria": [
            "Often loses temper",
            "Often touchy or easily annoyed",
            "Often angry and resentful",
            "Often argues with authority figures",
            "Often deliberately annoys others",
        ],
    },
    {
        "note": (
            "Patient presents with chronic pain in multiple body sites. "
            "Pain has lasted for 2 years without clear medical explanation. "
            "Excessive thoughts about seriousness of pain symptoms. "
            "High level of anxiety about health. "
            "Persistent preoccupation with symptoms affecting daily functioning."
        ),
        "criteria": [
            "One or more somatic symptoms that are distressing",
            "Excessive thoughts about seriousness of symptoms",
            "Persistently high level of anxiety about health",
            "Symptoms are persistently present (more than 6 months)",
            "Excessive time and energy devoted to symptoms",
        ],
    },
    {
        "note": (
            "Patient reports flashbacks of childhood abuse occurring several times weekly. "
            "Active avoidance of reminders of traumatic events. "
            "Negative beliefs about self developed since trauma. "
            "Persistent inability to experience positive emotions. "
            "Duration of symptoms is 14 months. Blames self for trauma."
        ),
        "criteria": [
            "Recurrent intrusive memories of traumatic event",
            "Efforts to avoid memories, thoughts, or feelings",
            "Persistent negative emotional state",
            "Persistent negative beliefs about oneself",
            "Persistent blame of self or others for traumatic event",
        ],
    },
    {
        "note": (
            "Patient displays pervasive distrust and suspicion of others. "
            "Reads hidden demeaning meanings into benign remarks. "
            "Reluctant to confide in clinician. "
            "Perceives attacks on character not apparent to others. "
            "Recurrently suspects fidelity of spouse without justification."
        ),
        "criteria": [
            "Suspects others are exploiting or deceiving them",
            "Reads hidden meanings into benign remarks",
            "Persistently bears grudges",
            "Reluctant to confide in others due to fear",
            "Perceives attacks on character not apparent to others",
        ],
    },
    {
        "note": (
            "Patient exhibits intense and unstable relationships. "
            "Frantic efforts to avoid abandonment noted. "
            "Identity disturbance with unstable self-image. "
            "Impulsivity in spending and substance use. "
            "Recurrent suicidal gestures in past. Chronic feelings of emptiness."
        ),
        "criteria": [
            "Frantic efforts to avoid real or imagined abandonment",
            "Pattern of unstable and intense relationships",
            "Identity disturbance",
            "Impulsivity in at least two areas",
            "Chronic feelings of emptiness",
        ],
    },
    {
        "note": (
            "Patient presents with lack of interest in social relationships. "
            "Prefers solitary activities consistently. "
            "Little interest in sexual experiences. "
            "Takes pleasure in few activities. "
            "Lacks close friends other than family. Appears indifferent to praise or criticism."
        ),
        "criteria": [
            "Neither desires nor enjoys close relationships",
            "Almost always chooses solitary activities",
            "Little interest in sexual experiences with another person",
            "Takes pleasure in few activities",
            "Lacks close friends or confidants",
        ],
    },
    {
        "note": (
            "Patient reports periods of elevated mood alternating with depression. "
            "During elevated periods, decreased need for sleep noted. "
            "Increased productivity and talkativeness during highs. "
            "Depressive periods include low energy and hopelessness. "
            "Pattern present for 18 months. No periods of normal mood longer than 2 months."
        ),
        "criteria": [
            "Numerous periods with hypomanic symptoms",
            "Numerous periods with depressive symptoms",
            "Hypomanic and depressive periods present for at least 2 years",
            "Never without symptoms for more than 2 months at a time",
            "Symptoms cause significant distress or impairment",
        ],
    },
    {
        "note": (
            "Patient reports difficulty falling asleep most nights for 6 months. "
            "Sleep latency typically 60-90 minutes. "
            "Daytime fatigue and irritability present. "
            "Worries about sleep interfering with ability to fall asleep. "
            "Occurs at least 3 nights per week. Adequate sleep opportunity provided."
        ),
        "criteria": [
            "Predominant complaint of difficulty initiating sleep",
            "Sleep disturbance occurs at least 3 nights per week",
            "Sleep disturbance present for at least 3 months",
            "Sleep difficulty causes significant distress or impairment",
            "Adequate opportunity for sleep exists",
        ],
    },
    {
        "note": (
            "Patient presents with restricted food intake for 8 months. "
            "Current BMI is 16.5 (significantly low). "
            "Intense fear of gaining weight despite being underweight. "
            "Body image disturbance noted - sees self as overweight. "
            "Persistent behavior interfering with weight gain. Denies problem."
        ),
        "criteria": [
            "Restriction of energy intake leading to significantly low body weight",
            "Intense fear of gaining weight or becoming fat",
            "Disturbance in the way body weight or shape is experienced",
            "Persistent behavior that interferes with weight gain",
            "Lack of recognition of seriousness of low body weight",
        ],
    },
    {
        "note": (
            "Patient reports excessive alcohol consumption daily for 3 years. "
            "Unsuccessful efforts to cut down noted. "
            "Great deal of time spent obtaining and using alcohol. "
            "Continued use despite knowledge of physical problems. "
            "Tolerance has developed - needs increasing amounts. Withdrawal symptoms when stopping."
        ),
        "criteria": [
            "Alcohol taken in larger amounts over longer period than intended",
            "Persistent desire or unsuccessful efforts to cut down",
            "Great deal of time spent in activities to obtain alcohol",
            "Continued use despite knowledge of physical or psychological problems",
            "Tolerance (need for increased amounts)",
        ],
    },
    {
        "note": (
            "Elderly patient presents with progressive memory decline over 18 months. "
            "Difficulty learning new information noted. "
            "Gets lost in familiar neighborhoods. "
            "Impaired judgment in financial decisions. "
            "Language difficulties with word-finding. Represents decline from previous functioning."
        ),
        "criteria": [
            "Evidence of significant cognitive decline in memory",
            "Cognitive deficits interfere with independence",
            "Deficits do not occur exclusively in context of delirium",
            "Represents decline from previous level of functioning",
            "Learning and recall of new information is impaired",
        ],
    },
    {
        "note": (
            "Patient reports urges to set fires occurring multiple times. "
            "Has set several fires in past year. "
            "Fascination with fire and its contexts noted. "
            "Tension before fire-setting and relief afterward reported. "
            "Pleasure and gratification from fire-setting. Not done for monetary gain."
        ),
        "criteria": [
            "Deliberate and purposeful fire setting on more than one occasion",
            "Tension or affective arousal before fire setting",
            "Fascination with, interest in, curiosity about fire",
            "Pleasure, gratification, or relief when setting fires",
            "Fire setting not done for monetary gain or sabotage",
        ],
    },
    {
        "note": (
            "Child patient presents with repetitive motor movements. "
            "Eye blinking tics occur multiple times daily for 10 months. "
            "Motor tics include head jerking and shoulder shrugging. "
            "Tics occur many times per day nearly every day. "
            "Onset was at age 7, currently age 8. Waxing and waning pattern observed."
        ),
        "criteria": [
            "Multiple motor tics present",
            "Tics occur many times a day",
            "Tics present for more than one year",
            "Onset before age 18",
            "Tics not attributable to physiological effects of substance",
        ],
    },
    {
        "note": (
            "Patient displays excessive emotionality and attention-seeking. "
            "Uncomfortable when not center of attention. "
            "Interaction style is inappropriately sexually seductive. "
            "Rapidly shifting and shallow expression of emotions. "
            "Uses physical appearance to draw attention. Speech is impressionistic and lacks detail."
        ),
        "criteria": [
            "Uncomfortable when not center of attention",
            "Interaction characterized by inappropriate sexual seductiveness",
            "Displays rapidly shifting and shallow expression of emotions",
            "Uses physical appearance to draw attention",
            "Style of speech that is excessively impressionistic",
        ],
    },
    {
        "note": (
            "Patient reports stealing items without need or monetary value. "
            "Increasing tension before theft and relief during act. "
            "Has been caught shoplifting three times in past 6 months. "
            "Items stolen are not needed and often given away. "
            "Not done in anger or vengeance. Feels guilty afterward but continues behavior."
        ),
        "criteria": [
            "Recurrent failure to resist impulses to steal",
            "Increasing tension before committing theft",
            "Pleasure or relief at time of committing theft",
            "Stealing not committed to express anger or vengeance",
            "Stealing not done for personal use or monetary value",
        ],
    },
]


def build_demo_samples(num_samples: int = 20, seed: int = 42) -> List[Sample]:
    """Generate synthetic clinical note samples.

    Args:
        num_samples: Number of samples to generate (default 20).
        seed: Random seed for reproducibility (default 42).

    Returns:
        List of Sample objects with clinical notes and criteria.
    """
    random.seed(seed)
    samples = []

    # Use all predefined notes and sample additional if needed
    notes_to_use = CLINICAL_NOTES.copy()
    if num_samples > len(CLINICAL_NOTES):
        # Duplicate and modify some notes for variation
        extra_needed = num_samples - len(CLINICAL_NOTES)
        notes_to_use.extend(random.choices(CLINICAL_NOTES, k=extra_needed))

    for idx, note_data in enumerate(notes_to_use[:num_samples]):
        sample = Sample(
            id=f"demo-note-{idx + 1:03d}",
            note_text=note_data["note"],
            criteria=note_data["criteria"],
            metadata={"source": "synthetic", "template_idx": idx % len(CLINICAL_NOTES)},
        )
        samples.append(sample)

    return samples


def main() -> None:
    """Generate demo training and test datasets."""
    # Generate training data (20 samples)
    train_output = Path("data/raw/demo_train.jsonl")
    train_output.parent.mkdir(parents=True, exist_ok=True)
    train_samples = build_demo_samples(num_samples=20, seed=42)

    with train_output.open("w", encoding="utf-8") as handle:
        for sample in train_samples:
            handle.write(sample.to_json() + "\n")
    print(f"Wrote {len(train_samples)} demo training samples to {train_output}")

    # Generate test data (5 samples using different seed)
    test_output = Path("data/raw/demo_test.jsonl")
    test_output.parent.mkdir(parents=True, exist_ok=True)
    test_samples = build_demo_samples(num_samples=5, seed=100)

    with test_output.open("w", encoding="utf-8") as handle:
        for sample in test_samples:
            handle.write(sample.to_json() + "\n")
    print(f"Wrote {len(test_samples)} demo test samples to {test_output}")

    # Also write to default location for backward compatibility
    legacy_output = Path("data/raw/train.jsonl")
    with legacy_output.open("w", encoding="utf-8") as handle:
        for sample in train_samples:
            handle.write(sample.to_json() + "\n")
    print(f"Wrote {len(train_samples)} samples to legacy location {legacy_output}")


if __name__ == "__main__":
    main()
