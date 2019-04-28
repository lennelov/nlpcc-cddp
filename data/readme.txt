************ Data for Cross-Domain Dependency Parsing ************

1. Sources & Format
	The training and test data are transformed from Chinese Treebank (CTB) 9.0 (https://catalog.ldc.upenn.edu/LDC2016T13) and UD_Chinese-CFL (https://github.com/UniversalDependencies/UD_Chinese-CFL).

	All annotations are in the UD format (http://universaldependencies.org/format.html).

	Note:
		The UD_Chinese-CFL corpus is based on essays written by learners of Mandarin Chinese as a foreign language (L2).

2. Distributions of these sentences are as follows:

	Training:
			Newswire -- 230
			Magazine articles -- 230
			Broadcast news -- 230
			Broadcast conversations -- 230
			Weblogs -- 230
			Discussion forums -- 230
			SMS/Chat messages -- 230
			Conversational speech -- 230
			UD_Chinese-CFL -- 160
		Total -- 2000 sentences
	Test:
			Newswire -- 115
			Magazine articles -- 115
			Broadcast news -- 115
			Broadcast conversations -- 115
			Weblogs -- 115
			Discussion forums -- 115
			SMS/Chat messages -- 115
			Conversational speech -- 115
			UD_Chinese-CFL -- 80
		Total -- 1000 sentences

3. References

	Xue, Naiwen, et al. "The Penn Chinese TreeBank: Phrase structure annotation of a large corpus." Natural language engineering 11.2 (2005): 207-238.

	John Lee, Herman Leung, Keying Li. 2017. Towards Universal Dependencies for Learner Chinese. In Proc. Workshop on Universal Dependencies.
