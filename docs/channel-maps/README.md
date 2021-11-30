# channel mapping details with spike gadgets

## visualizing on the acquisition system

The channels on Trodes are given nTrodeIDs 1-64. I have ordered these IDs so that it looks like the following on the two shanks for the 64 channel probe:

		1		2				33		34
	3		4		5		35		36		37
		6		7				38		39
	8		9		10		40		41		42
		11		12				43		44
	13		14		15		45		46		47
		16		17				48		49
	18		19		20		50		51		52
		21		22				53		54
	23		24		25		55		56		57
		26		27				58		59
	28		29		30		60		61		62
		31		32				63		64
				
		    left shank					    right shank

These nTrodeIDs had to be mapped from the probe channels to the Neuronexus adapter channels to the SpikeGadgets headstage channels. These calculations can be found in this folder under SpikeGadgets_A2x32Poly5_Mapping_210327_RigC.csv.

## getting the correct channels from the singer lab preprocessed data structures

The preprocessed data is saved into folders with channel numbers. These numbers are the hardware channels, NOT the nTrodeIDs. So to get the correct order of channels on the probe, you can use the SpikeGadgets_A2x32Poly5_Mapping_210327_RigC.csv. For example, since nTrodeID 1 is the top left corner, the folder labeled 28 would have the data from the top left corner of the probe. 