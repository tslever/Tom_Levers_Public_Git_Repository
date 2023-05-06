package midi_parsing_utilities;

import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiMessage;
import javax.sound.midi.Sequence;
import javax.sound.midi.ShortMessage;
import javax.sound.midi.Track;
import java.util.ArrayList;
import java.util.TreeMap;

public class MidiParser {

	private static int NOTE_ON = 0x90;
	private static int NOTE_OFF = 0x80;
	
	public static TreeMap<Integer, ArrayList<TickCommandKeyAndVelocity>>
		getTreeMapOfChannelsAndArrayListsFrom(Sequence sequence) {
		
		int l;
		MidiEvent midiEvent;
		MidiMessage midiMessage;
		ShortMessage shortMessage;
		TreeMap<Integer, ArrayList<TickCommandKeyAndVelocity>> treeMapOfChannelsAndArrayLists =
			new TreeMap<Integer, ArrayList<TickCommandKeyAndVelocity>>();
		int pianoKey;
		int velocity;
		int[] arrayOfCommandsKeysAndVelocities = new int[3];
		
		for (Track track : sequence.getTracks()) {
			
			for (l = 0; l < track.size(); l++) {
				
				midiEvent = track.get(l);
				midiMessage = midiEvent.getMessage();
				
				if (midiMessage instanceof ShortMessage) {
					
					shortMessage = (ShortMessage)midiMessage;
					
					if (!treeMapOfChannelsAndArrayLists.containsKey(shortMessage.getChannel())) {
						treeMapOfChannelsAndArrayLists.put(
							shortMessage.getChannel(), new ArrayList<TickCommandKeyAndVelocity>()
						);
					}
					
					if ((shortMessage.getCommand() == NOTE_ON) || (shortMessage.getCommand() == NOTE_OFF)) {
						
						pianoKey = shortMessage.getData1();
						velocity = shortMessage.getData2();
						
						arrayOfCommandsKeysAndVelocities[0] = shortMessage.getCommand();
						arrayOfCommandsKeysAndVelocities[1] = pianoKey;
						arrayOfCommandsKeysAndVelocities[2] = velocity;
						
						treeMapOfChannelsAndArrayLists.get(shortMessage.getChannel()).add(
							new TickCommandKeyAndVelocity(
								(int)midiEvent.getTick(), shortMessage.getCommand(), pianoKey, velocity
							)
						);
						
					}
					
				}
				
			}
			
		}
		
		return treeMapOfChannelsAndArrayLists;
	}
	
	public static int getNoteOn() {
		return NOTE_ON;
	}
	
	public static int getNoteOff() {
		return NOTE_OFF;
	}
}
