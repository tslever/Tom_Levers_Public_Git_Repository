package converter_utilities;


import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

import converter_utilities.InputParsingUtilities;


class InputParsingUtilitiesTest {

	@Test
	void testCheckNumberOfInputsIn() {
		
		try {
			InputParsingUtilities.checkNumberOfInputsIn(new String[]{"abc"});
		}
		catch (NotTwoInputsException e) {
			System.out.println(
				"testCheckNumberOfInputsIn has successfully checked number of inputs in String array with 'abc'.");
			return;
		}
		fail("testCheckNumberOfInputsIn has unsuccessfully checked number of inputs in String array with 'abc'.");
	}

	@Test
	void testGetContentsOfFile() {
		fail("Not yet implemented");
	}

}
