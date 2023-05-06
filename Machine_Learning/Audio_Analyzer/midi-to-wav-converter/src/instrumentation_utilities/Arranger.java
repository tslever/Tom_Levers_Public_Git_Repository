package instrumentation_utilities;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.TreeMap;

public class Arranger {
	
	public static TreeMap<Integer, TreeMap<Integer, double[]>> loadInstrumentSamples() throws FileNotFoundException, IOException {
		
		TreeMap<Integer, double[]> fluteSamples = new TreeMap<Integer, double[]>();
		fluteSamples.put(59, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.B3.stereo.trimmed.wav"));
		fluteSamples.put(60, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.C4.stereo.trimmed.wav"));
		fluteSamples.put(61, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.Db4.stereo.trimmed.wav"));
		fluteSamples.put(62, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.D4.stereo.trimmed.wav"));
		fluteSamples.put(63, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.Eb4.stereo.trimmed.wav"));
		fluteSamples.put(64, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.E4.stereo.trimmed.wav"));
		fluteSamples.put(65, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.F4.stereo.trimmed.wav"));
		fluteSamples.put(66, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.Gb4.stereo.trimmed.wav"));
		fluteSamples.put(67, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.G4.stereo.trimmed.wav"));
		fluteSamples.put(68, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.Ab4.stereo.trimmed.wav"));
		fluteSamples.put(69, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.A4.stereo.trimmed.wav"));
		fluteSamples.put(70, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.Bb4.stereo.trimmed.wav"));
		fluteSamples.put(71, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.B4.stereo.trimmed.wav"));
		fluteSamples.put(72, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.C5.stereo.trimmed.wav"));
		fluteSamples.put(73, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.Db5.stereo.trimmed.wav"));
		fluteSamples.put(74, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.D5.stereo.trimmed.wav"));
		fluteSamples.put(75, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.Eb5.stereo.trimmed.wav"));
		fluteSamples.put(76, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.E5.stereo.trimmed.wav"));
		fluteSamples.put(77, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.F5.stereo.trimmed.wav"));
		fluteSamples.put(78, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.Gb5.stereo.trimmed.wav"));
		fluteSamples.put(79, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.G5.stereo.trimmed.wav"));
		fluteSamples.put(80, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.Ab5.stereo.trimmed.wav"));
		fluteSamples.put(81, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.A5.stereo.trimmed.wav"));
		fluteSamples.put(82, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.Bb5.stereo.trimmed.wav"));
		fluteSamples.put(83, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.B5.stereo.trimmed.wav"));
		fluteSamples.put(84, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.C6.stereo.trimmed.wav"));
		fluteSamples.put(85, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.Db6.stereo.trimmed.wav"));
		fluteSamples.put(86, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.D6.stereo.trimmed.wav"));
		fluteSamples.put(87, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.Eb6.stereo.trimmed.wav"));
		fluteSamples.put(88, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.E6.stereo.trimmed.wav"));
		fluteSamples.put(89, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.F6.stereo.trimmed.wav"));
		fluteSamples.put(90, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.Gb6.stereo.trimmed.wav"));
		fluteSamples.put(91, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.G6.stereo.trimmed.wav"));
		fluteSamples.put(92, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.Ab6.stereo.trimmed.wav"));
		fluteSamples.put(93, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.A6.stereo.trimmed.wav"));
		fluteSamples.put(94, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.Bb6.stereo.trimmed.wav"));
		fluteSamples.put(95, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.B6.stereo.trimmed.wav"));
		fluteSamples.put(96, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.C7.stereo.trimmed.wav"));
		fluteSamples.put(97, readWavFile(
			"resources/Musical_Instrument_Samples/Flute/Flute.nonvib.ff.Db7.stereo.trimmed.wav"));
		
		TreeMap<Integer, TreeMap<Integer, double[]>> instrumentSamples =
			new TreeMap<Integer, TreeMap<Integer, double[]>>();
		instrumentSamples.put(0, fluteSamples);
		
		return instrumentSamples;
	}
	
	public static double[] readWavFile(String pathToWavFile) throws FileNotFoundException, IOException {
		
		FileInputStream fileInputStream = new FileInputStream(pathToWavFile); // throws FileNotFoundException
		
		byte[] subchunk2Size = new byte[4];
		fileInputStream.skip(40);
		fileInputStream.read(subchunk2Size, 0, 4);
		int numberOfBytesOfData = littleEndianByteArrayToInt(subchunk2Size);
		
		byte[] data = new byte[numberOfBytesOfData];
		fileInputStream.read(data, 0, numberOfBytesOfData);
		
		fileInputStream.close();
		
		double[] dataAsDoubleArray = new double[numberOfBytesOfData / 2];
		byte[] bytes = new byte[2];
		int i;
		for (i = 0; i < dataAsDoubleArray.length; i++) {
			bytes[0] = data[2*i];
			bytes[1] = data[2*i + 1];
			dataAsDoubleArray[i] = (double)littleEndianByteArrayToShort(bytes);
		}
		
		double maximum = -1.0;
		for (i = 0; i < dataAsDoubleArray.length; i++) {
			if (dataAsDoubleArray[i] > maximum) maximum = dataAsDoubleArray[i];
		}
		
		for (i = 0; i < dataAsDoubleArray.length; i++) {
			dataAsDoubleArray[i] = dataAsDoubleArray[i] / maximum + 1.0;
		}
		
		return dataAsDoubleArray;
	}
	
	private static int littleEndianByteArrayToInt(byte[] byteArray) {
		return (byteArray[0] & 0xFF) |
			   ((byteArray[1] & 0xFF) << 8) |
			   ((byteArray[2] & 0xFF) << 16) |
			   ((byteArray[3] & 0xFF) << 24);
	}
	
	private static short littleEndianByteArrayToShort(byte[] byteArray) {
		return (short)((byteArray[0] & 0xFF) | ((byteArray[1] & 0xFF) << 8));
	}
}