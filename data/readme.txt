************ Cross-Domain Dependency Parsing Task ************

1. Task Description

	This project is about cross-domain dependency parsing.
	We provide 2000 and 1000 sentences as the training and test data respectively.
	These two parts of data have different distributions.
	Participants are expected to build a cross-domain model, and evaluate the performance on the test data.

2. Data format

	All annotations are in the CoNLL-U format (http://universaldependencies.org/format.html).

		2.1 Word lines containing the annotation of a word/token in 10 fields separated by single tab characters; see below.

			ID: Word index, integer starting at 1 for each new sentence.
			FORM: Word form or punctuation symbol.
			LEMMA: Lemma or stem of word form.
			UPOS: Universal part-of-speech tag.
			XPOS: Language-specific part-of-speech tag; underscore if not available.
			FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
			HEAD: Head of the current word, which is either a value of ID or zero (0).
			DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one; underscore if not available.
			DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
			MISC: Any other annotation.

			*In this task, we only provide the 'ID, FORM, UPOS, XPOS, HEAD' fields.

		2.2 Blank lines marking sentence boundaries.

3. Submission & Evaluation
	
	Please submit a .conllu file containing parsed trees of test sentences, named as 'test.out.conllu'.

	We use the standard unlabeled attachment score (UAS, percentage of words that receive correct heads) as the evaluation metric.