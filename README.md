<h2>Movie Listening Physiological Data (Cardiac and Respiratory Traces)</h2>
<h2>OpenFMRI Study Forrest dataset, task001 Forrest Gump Audio Movie in German (8 Segments)</h2>
<h2>Michael A. Casey - Dartmouth College, Data and Code: 2013 - 2019</h2>

<h2>Instructions</h2>
Place the files in this repository in studyforrest/forrest_gump <h2>

<h2>The Physiological Data:</h2>

This dataset contains files for 20 subjects, 8 runs per subject. This folder contains only the physiological data.

Location sub[ID]/physio/task001_run00[1-8]/physio.txt.gz 
    
Physiological data were:

- truncated to start with the first MRI trigger pulse and to end one volume acquisition duration after the last trig-ger pulse. Data are provided in a four-column (MRI trigger, respira-tory trace, cardiac trace and oxygen saturation), space-delimited text file for each run. A log file of the automated conversion procedure is provided in the same directory (conversion.log). Sampling rate for the majority of all participants is 200 Hz.

- recorded simultaneously with the fMRI data acquisition using a custom setup25 and in-house recording software (written in Python). A Nonin 8600 FO pulse oxymeter (Nonin Medical, Inc, Plymouth, MN, USA) was used to measure cardiac trace and oxygen saturation, and a Siemens respiratory belt connected to a pressure sensor (Honeywell 40PC001B1A) captured the respiratory trace. Analog signals were digitized (12 bit analog digital converter, National Instruments USB-6008) at a sampling rate of 200 Hz. The digitizer also logged the volume acquisition trigger pulse emitted by the MRI scanner.

- down-sampled to <b>100 Hz</b> and truncated to start with the first MRI trigger pulse and to end one volume acquisition duration after the last trigger pulse. Data are provided in a four-column (MRI trigger, respiratory trace, cardiac trace and oxygen saturation), space-delimited text file for each movie segment. A log file of the automated conversion procedure is provided in the same directory (conversion.log).

<h2>References:</h2>

Michael Casey, Music of the 7Ts: Predicting and Decoding Multivoxel fMRI Responses with Acoustic, Schematic, and Categorical Music Features, <i>Frontiers in Psychology: Cognition</i>, 2017

Michael Hanke, Richard Dinga, Christian Häusler, J. Swaroop Guntupalli, Michael Casey, Falko R. Kaule, Jörg Stadler, High-resolution 7-Tesla fMRI data on the perception of musical genres, <i>F1000 Research</i>, 2015.

Michael Hanke, Florian J. Baumgartner, Pierre Ibe, Falko R. Kaule, Stefan Pollmann, Oliver Speck, Wolf Zinke & Jörg Stadler, A high-resolution 7-Tesla fMRI dataset from complex natural stimulation with an audio movie, <i>Scientific Data</i>, Volume 1, Article number: 140003 (2014) <br />

