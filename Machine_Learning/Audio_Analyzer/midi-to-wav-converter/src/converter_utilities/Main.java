package converter_utilities;


import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.TreeMap;
import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiSystem;
import javax.sound.midi.Sequence;

import midi_parsing_utilities.MidiParser;
import midi_parsing_utilities.TickCommandKeyAndVelocity;
import wav_assembling_utilities.StartTickHasNoCorrespondingEndTickException;
import wav_assembling_utilities.WavAssembler;


public class Main {
	
	public static void main(String[] args)
		throws NotOneInputException,
			   IOException,
			   InvalidMidiDataException,
			   StartTickHasNoCorrespondingEndTickException {
		
		InputParsingUtilities.checkNumberOfInputsIn(args); // throws NotOneInputException
		
		String filename = args[0];
		File file = new File(filename);
		Sequence sequence = MidiSystem.getSequence(file); // throws IOException, InvalidMidiDataException
		
		TreeMap<Integer, ArrayList<TickCommandKeyAndVelocity>> treeMapOfChannelsAndArrayLists =
			MidiParser.getTreeMapOfChannelsAndArrayListsFrom(sequence);
		
		short[] timeSeries = WavAssembler.assembleTimeSeriesBasedOn(sequence, treeMapOfChannelsAndArrayLists);

		WavAssembler.writeToWavFile(timeSeries, filename);
		
	}
}