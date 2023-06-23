## notes on electrophysiology recordings and noise

If using this dataset, you should be aware of some noise issues I observed/encountered during these recordings.
In all cases, this noise was attempted to be controlled during the recording, but there are some cases where it may still be present in the collected data.

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
Sometimes the manipulators or manipulator controller was turned off during a recording to try to control noise. This seemed to cause a high frequency signal that looked
like a 60 Hz wave but was much higher frequency and really only visible in phy when looking at the waveforms (very tiny triangle wave)
This was controlled for by not turning off the manipulators or control box during a recording. I realized it after I used this as a denoising technique, so it is present
in some recordings but should be noted in the recording logs.

### texting noise
When texts were received/sent on my phone or the phone was held high enough above the shielding, this could cause a short burst of high frequency square wave activity.
These were observed in the spike sorting output and could contaminate the single unit activity. 
They were removed with manual curation through phy.
This was discovered later in the recording process so it was present in several recordings but controlled for by putting the phone on the rig rack so that the faraday
cage was blocking the electrodes from the phone.