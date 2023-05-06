package converter_utilities;

public class InputParsingUtilities {

	public static void checkNumberOfInputsIn(String[] args) throws NotOneInputException {
		
		if (args.length != 1) {
			throw new NotOneInputException("Length of argument string is not equal to 1.");
		}
		
	}
}
