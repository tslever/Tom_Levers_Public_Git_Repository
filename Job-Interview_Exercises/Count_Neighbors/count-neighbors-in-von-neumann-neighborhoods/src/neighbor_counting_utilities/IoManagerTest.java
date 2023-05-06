package neighbor_counting_utilities;


import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;


class IoManagerTest {

	@Test
	void testGetHeightWidthDesiredNumberOfSettlersAndRange() {

		// Test getHeightWidthDesiredNumberOfSettlersAndRange for a valid set of arguments.
		/*try {
			IoManager.getHeightWidthDesiredNumberOfSettlersAndRange(new String[] {"9", "16", "3", "2"});
		}
		catch (Exception e) {
			e.printStackTrace();
		}*/

		// Test getHeightWidthDesiredNumberOfSettlersAndRange for one too many arguments.
		/*try {
			IoManager.getHeightWidthDesiredNumberOfSettlersAndRange(new String[] {"9", "16", "3", "2", "0"});
		}
		catch (Exception e) {
			e.printStackTrace();
		}*/
		
		// Test getHeightWidthDesiredNumberOfSettlersAndRange for a non-integer argument.
		/*try {
			IoManager.getHeightWidthDesiredNumberOfSettlersAndRange(new String[] {"9.0", "16", "3", "2"});
		}
		catch (Exception e) {
			e.printStackTrace();
		}*/
		
		// Test getHeightWidthDesiredNumberOfSettlersAndRange for a out-of-range argument.
		/*try {
			IoManager.getHeightWidthDesiredNumberOfSettlersAndRange(new String[] {"9", "16", "3", "2"});
		}
		catch (Exception e) {
			e.printStackTrace();
		}*/
		
		// Test getHeightWidthDesiredNumberOfSettlersAndRange for world area too high.
		/*try {
			IoManager.getHeightWidthDesiredNumberOfSettlersAndRange(new String[] {"2147483647", "2", "3", "2"});
		}
		catch (Exception e) {
			e.printStackTrace();
		}*/

	}

}
