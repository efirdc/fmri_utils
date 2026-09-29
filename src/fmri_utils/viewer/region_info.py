"""Descriptions and related work for atlas regions, for the viewer's region info box.

``harvard_oxford_cortical()`` returns ``{region name: {"abbrev", "description",
"related": [{"citation", "url"}], "attribution"}}`` keyed by the exact names in FSL's
``HarvardOxford-Cortical.xml``. Descriptions say where the region is, which
landmarks bound it, and what it is usually associated with; they are summaries
of standard neuroanatomy, not claims from any one analysis. Related-work links
are Google Scholar searches for the paper's exact title, so a link cannot point
at the wrong paper through a mistyped identifier.

Pass the result itself, keyed by region name, as ``ViewerSpec.region_info`` (or
the manifest key ``region_info``); the page strips a "Left "/"Right " prefix
before looking a region up.
"""

from __future__ import annotations

from urllib.parse import quote_plus

# (short citation, exact title). One place, so a reference is typed once.
_WORKS = {
    "binder2009": ("Binder et al. (2009) Cereb Cortex 19:2767",
                   "Where is the semantic system? A critical review and meta-analysis of 120 functional neuroimaging studies"),
    "lambonralph2017": ("Lambon Ralph et al. (2017) Nat Rev Neurosci 18:42",
                        "The neural and computational bases of semantic cognition"),
    "hickok2007": ("Hickok & Poeppel (2007) Nat Rev Neurosci 8:393",
                   "The cortical organization of speech processing"),
    "fedorenko2010": ("Fedorenko et al. (2010) J Neurophysiol 104:1177",
                      "New method for fMRI investigations of language: defining ROIs functionally in individual subjects"),
    "hagoort2005": ("Hagoort (2005) Trends Cogn Sci 9:416",
                    "On Broca, brain, and binding: a new framework"),
    "price2012": ("Price (2012) NeuroImage 62:816",
                  "A review and synthesis of the first 20 years of PET and fMRI studies of heard speech, spoken language and reading"),
    "scott2000": ("Scott et al. (2000) Brain 123:2400",
                  "Identification of a pathway for intelligible speech in the left temporal lobe"),
    "morosan2001": ("Morosan et al. (2001) NeuroImage 13:684",
                    "Human primary auditory cortex: cytoarchitectonic subdivisions and mapping into a spatial reference system"),
    "griffiths2002": ("Griffiths & Warren (2002) Trends Neurosci 25:348",
                      "The planum temporale as a computational hub"),
    "saxe2003": ("Saxe & Kanwisher (2003) NeuroImage 19:1835",
                 "People thinking about thinking people: the role of the temporo-parietal junction in theory of mind"),
    "amodio2006": ("Amodio & Frith (2006) Nat Rev Neurosci 7:268",
                   "Meeting of minds: the medial frontal cortex and social cognition"),
    "deen2015": ("Deen et al. (2015) Cereb Cortex 25:4596",
                 "Functional organization of social perception and cognition in the superior temporal sulcus"),
    "olson2007": ("Olson, Plotzker & Ezzyat (2007) Brain 130:1718",
                  "The enigmatic temporal pole: a review of findings on social and emotional processing"),
    "seghier2013": ("Seghier (2013) Neuroscientist 19:43",
                    "The angular gyrus: multiple functions and multiple subdivisions"),
    "cavanna2006": ("Cavanna & Trimble (2006) Brain 129:564",
                    "The precuneus: a review of its functional anatomy and behavioural correlates"),
    "leech2014": ("Leech & Sharp (2014) Brain 137:12",
                  "The role of the posterior cingulate cortex in cognition and disease"),
    "bush2000": ("Bush, Luu & Posner (2000) Trends Cogn Sci 4:215",
                 "Cognitive and emotional influences in anterior cingulate cortex"),
    "ridderinkhof2004": ("Ridderinkhof et al. (2004) Science 306:443",
                         "The role of the medial frontal cortex in cognitive control"),
    "nachev2008": ("Nachev, Kennard & Husain (2008) Nat Rev Neurosci 9:856",
                   "Functional role of the supplementary and pre-supplementary motor areas"),
    "ramnani2004": ("Ramnani & Owen (2004) Nat Rev Neurosci 5:184",
                    "Anterior prefrontal cortex: insights into function from anatomy and neuroimaging"),
    "kringelbach2005": ("Kringelbach (2005) Nat Rev Neurosci 6:691",
                        "The human orbitofrontal cortex: linking reward to hedonic experience"),
    "duncan2010": ("Duncan (2010) Trends Cogn Sci 14:172",
                   "The multiple-demand (MD) system of the primate brain: mental programs for intelligent behaviour"),
    "corbetta2002": ("Corbetta & Shulman (2002) Nat Rev Neurosci 3:201",
                     "Control of goal-directed and stimulus-driven attention in the brain"),
    "craig2009": ("Craig (2009) Nat Rev Neurosci 10:59",
                  "How do you feel--now? The anterior insula and human awareness"),
    "mayberg2005": ("Mayberg et al. (2005) Neuron 45:651",
                    "Deep brain stimulation for treatment-resistant depression"),
    "penfield1937": ("Penfield & Boldrey (1937) Brain 60:389",
                     "Somatic motor and sensory representation in the cerebral cortex of man as studied by electrical stimulation"),
    "brown2008": ("Brown, Ngan & Liotti (2008) Cereb Cortex 18:837",
                  "A larynx area in the human motor cortex"),
    "eickhoff2006": ("Eickhoff et al. (2006) Cereb Cortex 16:254",
                     "The human parietal operculum. I. Cytoarchitectonic mapping of subdivisions"),
    "culham2001": ("Culham & Kanwisher (2001) Curr Opin Neurobiol 11:157",
                   "Neuroimaging of cognitive functions in human parietal cortex"),
    "grillspector2001": ("Grill-Spector, Kourtzi & Kanwisher (2001) Vision Res 41:1409",
                         "The lateral occipital complex and its role in object recognition"),
    "tootell1995": ("Tootell et al. (1995) J Neurosci 15:3215",
                    "Functional analysis of human MT/V5 using magnetic resonance imaging"),
    "downing2001": ("Downing et al. (2001) Science 293:2470",
                    "A cortical area selective for visual processing of the human body"),
    "wandell2007": ("Wandell, Dumoulin & Brewer (2007) Neuron 56:366",
                    "Visual field maps in human cortex"),
    "grillspector2014": ("Grill-Spector & Weiner (2014) Nat Rev Neurosci 15:536",
                         "The functional architecture of the ventral temporal cortex and its role in categorization"),
    "kanwisher1997": ("Kanwisher, McDermott & Chun (1997) J Neurosci 17:4302",
                      "The fusiform face area: a module in human extrastriate cortex specialized for face perception"),
    "dehaene2011": ("Dehaene & Cohen (2011) Trends Cogn Sci 15:254",
                    "The unique role of the visual word form area in reading"),
    "epstein1998": ("Epstein & Kanwisher (1998) Nature 392:598",
                    "A cortical representation of the local visual environment"),
    "aminoff2013": ("Aminoff, Kveraga & Bar (2013) Trends Cogn Sci 17:379",
                    "The role of the parahippocampal cortex in cognition"),
    "squire2004": ("Squire, Stark & Clark (2004) Annu Rev Neurosci 27:279",
                   "The medial temporal lobe"),
    "dufour2013": ("Dufour et al. (2013) PLoS ONE 8:e75468",
                   "Similar brain activation during false belief tasks in a large sample of adults "
                   "with and without autism"),
}


def _related(*keys: str) -> list[dict]:
    out = []
    for key in keys:
        citation, title = _WORKS[key]
        out.append({"citation": f"{citation}. {title}.",
                    "url": "https://scholar.google.com/scholar?q=" + quote_plus(f'"{title}"')})
    return out


# name: (abbreviation, description, related-work keys)
_CORTICAL = {
    "Frontal Pole": ("FP",
        "The most anterior prefrontal cortex (roughly Brodmann area 10), on both the lateral and "
        "medial surfaces. Associated with holding goals in abeyance, relational reasoning and "
        "prospection; its medial part joins medial prefrontal cortex, associated with "
        "self-referential and social judgement.", ("ramnani2004",)),
    "Insular Cortex": ("Ins",
        "Cortex buried in the lateral sulcus beneath the opercula. The anterior insula is "
        "associated with interoception, awareness of bodily and emotional states and salience; "
        "the posterior insula with somatosensory and pain processing.", ("craig2009",)),
    "Superior Frontal Gyrus": ("SFG",
        "The dorsal frontal gyrus, running from the frontal pole back to the precentral gyrus and "
        "over the midline onto the medial surface. Its medial part includes the "
        "pre-supplementary motor area and dorsomedial prefrontal cortex (self-referential and "
        "social cognition); the lateral part borders the frontal eye field and joins working "
        "memory and attention networks. Harvard-Oxford does not separate these.",
        ("nachev2008", "amodio2006")),
    "Middle Frontal Gyrus": ("MFG",
        "Lateral frontal cortex between the superior and inferior frontal sulci, dorsolateral "
        "prefrontal cortex. Associated with working memory and cognitive control (the "
        "multiple-demand network); the frontal eye field lies at its posterior end.",
        ("duncan2010", "corbetta2002")),
    "Inferior Frontal Gyrus, pars triangularis": ("IFGt",
        "The triangular part of the inferior frontal gyrus (about Brodmann area 45), the anterior "
        "half of classical Broca's area. Consistently engaged by sentence comprehension and "
        "production, semantic selection and syntactic unification; a core region of the "
        "left-lateralised language network.", ("hagoort2005", "fedorenko2010")),
    "Inferior Frontal Gyrus, pars opercularis": ("IFGo",
        "The opercular part of the inferior frontal gyrus (about Brodmann area 44), the posterior "
        "half of classical Broca's area, just anterior to ventral premotor cortex. Associated "
        "with phonological and syntactic processing and speech motor planning (the dorsal "
        "stream).", ("hagoort2005", "hickok2007")),
    "Precentral Gyrus": ("PreCG",
        "Primary motor cortex (Brodmann area 4) and adjacent premotor cortex, anterior to the "
        "central sulcus, laid out somatotopically from the leg at the vertex to the face, larynx "
        "and tongue near the lateral sulcus. Ventral parts are active during speech "
        "articulation and often in speech perception.", ("penfield1937", "brown2008")),
    "Temporal Pole": ("TP",
        "The anterior tip of the temporal lobe. Associated with semantic memory (the "
        "anterior-temporal semantic hub), person knowledge, and social and emotional "
        "processing.",
        ("lambonralph2017", "olson2007")),
    "Superior Temporal Gyrus, anterior division": ("aSTG",
        "Lateral superior temporal gyrus anterior to Heschl's gyrus. Auditory association cortex "
        "on the ventral stream for intelligible speech, responding more to sentences than to "
        "acoustically matched non-speech.", ("scott2000", "hickok2007")),
    "Superior Temporal Gyrus, posterior division": ("pSTG",
        "Lateral superior temporal gyrus at and behind Heschl's gyrus, including Wernicke's area "
        "in the classical sense. Associated with phonological processing of speech; its border "
        "with the superior temporal sulcus responds to voices, faces, biological motion and "
        "other social stimuli.", ("hickok2007", "deen2015")),
    "Middle Temporal Gyrus, anterior division": ("aMTG",
        "Anterior middle temporal gyrus. Associated with lexical and conceptual semantics and "
        "sentence-level meaning, together with the temporal pole.",
        ("binder2009", "lambonralph2017")),
    "Middle Temporal Gyrus, posterior division": ("pMTG",
        "Posterior middle temporal gyrus. Associated with lexical-semantic access and the "
        "mapping from sound to meaning (the ventral stream's lexical interface), and with "
        "controlled semantic retrieval.", ("hickok2007", "binder2009")),
    "Middle Temporal Gyrus, temporooccipital part": ("toMTG",
        "Where the middle temporal gyrus meets occipital cortex. Contains or neighbours the "
        "motion area MT/V5 and the extrastriate body area; also engaged by action and event "
        "semantics.", ("tootell1995", "downing2001", "binder2009")),
    "Inferior Temporal Gyrus, anterior division": ("aITG",
        "Anterior inferior temporal gyrus on the ventrolateral temporal surface. Part of the "
        "anterior temporal semantic system; high-level object and person knowledge.",
        ("lambonralph2017",)),
    "Inferior Temporal Gyrus, posterior division": ("pITG",
        "Posterior inferior temporal gyrus. Ventral visual stream cortex for object recognition "
        "and category-selective responses, and a basal-temporal language area.",
        ("grillspector2014",)),
    "Inferior Temporal Gyrus, temporooccipital part": ("toITG",
        "Inferior temporal gyrus at the temporo-occipital junction, near the occipitotemporal "
        "sulcus. Ventral visual cortex; the visual word form area sits close by, on the "
        "fusiform side.", ("grillspector2014", "dehaene2011")),
    "Postcentral Gyrus": ("PostCG",
        "Primary somatosensory cortex (Brodmann areas 3, 1 and 2), behind the central sulcus, "
        "somatotopic like the motor strip in front of it.", ("penfield1937",)),
    "Superior Parietal Lobule": ("SPL",
        "Dorsal parietal cortex above the intraparietal sulcus. Associated with spatial "
        "attention, visuomotor coordination and reaching (the dorsal attention network).",
        ("culham2001", "corbetta2002")),
    "Supramarginal Gyrus, anterior division": ("aSMG",
        "Anterior inferior parietal lobule, capping the posterior end of the lateral sulcus. "
        "Associated with somatosensory integration, tool use and phonological working memory.",
        ("culham2001",)),
    "Supramarginal Gyrus, posterior division": ("pSMG",
        "Posterior supramarginal gyrus, the anterior part of the temporoparietal junction. "
        "Associated with phonological working memory, reorienting attention to salient events "
        "(the ventral attention network) and, particularly on the right, reasoning about "
        "others' mental states.", ("corbetta2002", "saxe2003")),
    "Angular Gyrus": ("AG",
        "Posterior inferior parietal lobule, capping the superior temporal sulcus. A multimodal "
        "region associated with semantic processing, reading and number, episodic retrieval, "
        "spatial attention and social cognition; a node of the default-mode network.",
        ("seghier2013", "binder2009")),
    "Lateral Occipital Cortex, superior division": ("sLOC",
        "Dorsal lateral occipital cortex, extending up and forward to the parietal lobe. A large "
        "Harvard-Oxford region: its posterior part is visual (object- and scene-selective "
        "cortex, including the occipital place area), while its anterior part reaches the "
        "intraparietal sulcus and the angular gyrus, taking in visuospatial and multimodal "
        "parietal cortex.",
        ("grillspector2001", "wandell2007")),
    "Lateral Occipital Cortex, inferior division": ("iLOC",
        "Ventral lateral occipital cortex: the lateral occipital complex for object shape, and "
        "nearby motion-selective MT+.", ("grillspector2001", "tootell1995")),
    "Intracalcarine Cortex": ("ICC",
        "Cortex within the calcarine sulcus: primary visual cortex (V1), retinotopically "
        "organised.", ("wandell2007",)),
    "Frontal Medial Cortex": ("FMC",
        "Ventromedial prefrontal cortex on the medial surface below the paracingulate gyrus. "
        "Associated with valuation, emotion regulation, and self-referential and social "
        "thought; a node of the default-mode network.", ("kringelbach2005", "amodio2006")),
    "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": ("SMA",
        "The supplementary and pre-supplementary motor areas on the dorsomedial surface in "
        "front of the paracentral lobule. Associated with initiating and sequencing movement, "
        "including speech.", ("nachev2008",)),
    "Subcallosal Cortex": ("SubC",
        "Ventromedial cortex beneath the genu of the corpus callosum (subgenual cingulate). "
        "Associated with mood and autonomic regulation.", ("mayberg2005",)),
    "Paracingulate Gyrus": ("PaCG",
        "Medial frontal cortex above the cingulate gyrus, where a paracingulate sulcus is "
        "present. Associated with performance monitoring and cognitive control and, more "
        "anteriorly, with self-referential and social cognition.",
        ("ridderinkhof2004", "amodio2006")),
    "Cingulate Gyrus, anterior division": ("aCG",
        "Anterior cingulate cortex. Associated with conflict monitoring and cognitive control "
        "(dorsal part) and emotion (ventral part).", ("bush2000",)),
    "Cingulate Gyrus, posterior division": ("pCG",
        "Posterior cingulate cortex, a central hub of the default-mode network. Associated with "
        "internally directed thought, autobiographical memory and self-referential processing, "
        "and with balancing internal against external attention.",
        ("leech2014",)),
    "Precuneous Cortex": ("PCun",
        "Medial parietal cortex in front of the parieto-occipital sulcus. Associated with "
        "episodic memory retrieval, visuospatial imagery, self-processing and perspective "
        "taking; part of the default-mode network.",
        ("cavanna2006",)),
    "Cuneal Cortex": ("Cun",
        "The cuneus, above the calcarine sulcus: early visual cortex for the lower visual field.",
        ("wandell2007",)),
    "Frontal Orbital Cortex": ("FOC",
        "Orbitofrontal cortex on the ventral frontal surface. Associated with reward value, "
        "outcome expectation and flexible choice.", ("kringelbach2005",)),
    "Parahippocampal Gyrus, anterior division": ("aPaHG",
        "Anterior parahippocampal gyrus, including entorhinal and perirhinal cortex. The "
        "cortical gateway to the hippocampus; memory and object familiarity.", ("squire2004",)),
    "Parahippocampal Gyrus, posterior division": ("pPaHG",
        "Posterior parahippocampal gyrus, containing the parahippocampal place area. "
        "Associated with scenes, spatial layout and contextual associations.",
        ("epstein1998", "aminoff2013")),
    "Lingual Gyrus": ("LG",
        "Medial occipital cortex below the calcarine sulcus: early visual cortex for the upper "
        "visual field, and ventral visual areas beyond it.", ("wandell2007",)),
    "Temporal Fusiform Cortex, anterior division": ("aTFus",
        "Anterior fusiform gyrus on the ventral temporal surface, part of the anterior temporal "
        "semantic system.", ("lambonralph2017",)),
    "Temporal Fusiform Cortex, posterior division": ("pTFus",
        "Posterior temporal fusiform gyrus: category-selective ventral visual cortex, including "
        "face-selective responses.", ("kanwisher1997", "grillspector2014")),
    "Temporal Occipital Fusiform Cortex": ("TOFus",
        "Fusiform gyrus at the temporo-occipital junction: face- and word-selective cortex, "
        "including the usual location of the visual word form area.",
        ("dehaene2011", "grillspector2014")),
    "Occipital Fusiform Gyrus": ("OFus",
        "Occipital fusiform cortex: intermediate ventral visual areas between early visual "
        "cortex and category-selective temporal cortex.", ("grillspector2014", "wandell2007")),
    "Frontal Operculum Cortex": ("FO",
        "Frontal operculum, between the inferior frontal gyrus and the anterior insula. "
        "Engaged in speech production and in sentence processing alongside Broca's area.",
        ("price2012",)),
    "Central Opercular Cortex": ("CO",
        "The operculum over the central sulcus: ventral sensorimotor cortex for the face, "
        "mouth and larynx, and secondary somatosensory cortex.",
        ("brown2008", "eickhoff2006")),
    "Parietal Operculum Cortex": ("PO",
        "The parietal operculum, the upper bank of the posterior lateral sulcus: secondary "
        "somatosensory cortex (OP1-OP4).", ("eickhoff2006",)),
    "Planum Polare": ("PP",
        "The supratemporal plane anterior to Heschl's gyrus. Auditory belt and parabelt "
        "cortex, responsive to complex sounds, music and speech.", ("scott2000", "hickok2007")),
    "Heschl's Gyrus (includes H1 and H2)": ("HG",
        "Heschl's gyrus on the supratemporal plane: primary auditory cortex (the auditory core, "
        "Te1), tonotopically organised.", ("morosan2001",)),
    "Planum Temporale": ("PT",
        "The supratemporal plane behind Heschl's gyrus. Auditory association cortex, larger on "
        "the left; associated with spectrotemporal analysis and the sensorimotor interface for "
        "speech (area Spt).", ("griffiths2002", "hickok2007")),
    "Supracalcarine Cortex": ("SCC",
        "Cortex just above the calcarine sulcus, at the upper edge of primary visual cortex.",
        ("wandell2007",)),
    "Occipital Pole": ("OP",
        "The posterior tip of the occipital lobe: early visual cortex representing the central "
        "visual field (foveal confluence).", ("wandell2007",)),
}


# Shown under every description in the info box.
ATTRIBUTION = "Description written by Claude Opus 5.5"


def harvard_oxford_cortical() -> dict:
    """Region info for Harvard-Oxford cortical, keyed by the FSL region names."""
    return {name: {"abbrev": abbrev, "description": description, "related": _related(*keys),
                   "attribution": ATTRIBUTION}
            for name, (abbrev, description, keys) in _CORTICAL.items()}


# The Saxe-lab theory-of-mind group parcels (false belief > false photograph,
# 462 participants; Dufour et al., 2013), under the names a viewer's atlas gives them.
_TOM_PARCELS = {
    "Right TPJ (ToM parcel)": ("RTPJ",
        "Right temporoparietal junction, where the posterior superior temporal sulcus meets the "
        "inferior parietal lobule. The most selective region in false-belief localisers for "
        "reasoning about other people's thoughts; also engaged by reorienting attention.",
        ("saxe2003", "dufour2013")),
    "Left TPJ (ToM parcel)": ("LTPJ",
        "Left temporoparietal junction, the left-hemisphere counterpart of the RTPJ parcel, "
        "spanning posterior angular gyrus and the surrounding lateral occipito-parietal cortex. "
        "Engaged by false-belief tasks and also by language and semantic processing.",
        ("saxe2003", "dufour2013")),
    "Precuneus (ToM parcel)": ("PC",
        "Precuneus and adjacent posterior cingulate cortex on the medial parietal surface. A "
        "core node of the default-mode network, engaged by false-belief tasks, episodic memory "
        "and self-referential thought.", ("cavanna2006", "dufour2013")),
    "Dorsal medial PFC (ToM parcel)": ("DMPFC",
        "Dorsal medial prefrontal cortex. Engaged when reasoning about other people's mental "
        "states and traits, and in self-referential judgement.", ("amodio2006", "dufour2013")),
    "Middle medial PFC (ToM parcel)": ("MMPFC",
        "Middle medial prefrontal cortex, between the dorsal and ventral medial parcels. Engaged "
        "by false-belief tasks and social and self-referential judgement.",
        ("amodio2006", "dufour2013")),
    "Ventral medial PFC (ToM parcel)": ("VMPFC",
        "Ventral medial prefrontal cortex. Associated with valuation and emotion as well as "
        "social judgement; the weakest of the seven parcels in false-belief localisers.",
        ("kringelbach2005", "dufour2013")),
    "Right STS (ToM parcel)": ("RSTS",
        "Right anterior superior temporal sulcus. Responds to social stimuli -- voices, faces, "
        "biological motion -- and to false-belief stories.", ("deen2015", "dufour2013")),
}


def saxe_tom_parcels() -> dict:
    """Region info for the Saxe-lab theory-of-mind parcels, keyed by their viewer names."""
    return {name: {"abbrev": abbrev, "description": description, "related": _related(*keys),
                   "attribution": ATTRIBUTION}
            for name, (abbrev, description, keys) in _TOM_PARCELS.items()}
