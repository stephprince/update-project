## notes on electrophysiology recordings and noise

If using this dataset, you should be aware of some noise issues I observered/encountered during these recordings.
In all cases, this noise was attempted to be controlled during the recording, but there are some cases where it may still be observed in the collected data.

### sharp movement noise
Sometimes I observed larger sharp deflections in the raw signal that look like spikes in the high pass filter view.
These can be identified by seeing spikes that occur simultaneously across all channels of the probes.
These occurred when the animal made sudden sharp movements or started running at an extremely high speed. 
They were controlled for during the recording by making a Kwik-Sil chamber, making sure the saline levels were good, and making sure the copper shielding was placed well.
They were attempted to be removed from the data by interpolating over outliers in the signal or separating noise clusters with manual curation in phy.

### EMG noise
Often times I observed small bursts/increases in the baseline activity that were often localized to the PFC probe. 
These looked like muscle related activity and they may have made some very small spikes harder to pick up at this time.
These occurred when the animal licked.

### lick detector noise
Sometimes, if the lick detector was not properly grounded or somehow the wires were in contact with the reward tubing, there was small, sharp deflections in the signal
These were related to licking activity and disappeared if the lick detector was unplugged.
These occurred when the animal made contact with the reward.
In most cases, these were controlled for by grounding the reward needle, but in a few recordings they may have persisted.

### Luigs and Neumann manipulator noise
Sometimes the manipulators or manipulator controller would emit a short burst of high frequency square wave activity.
The reasons/timing of why this would occurr were unclear, sometimes it would happen only once or twice in a recording, sometimes never.
These were observed in the spike sorting output and could contaminate the single unit activity. 
They were controlled for during the recording by grounding the manipulators or turning the manipulator control box off during the recording.
They were removed with manual curation through phy.