package Com.TSL.Settles_Of_Catan;

import java.util.Arrays;
import java.util.List;

public class NumberToken {

	public static final List<Integer> values = Arrays.asList(0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12);
	
	private final int value;
	
	public NumberToken(int valueToUse) {
		value = valueToUse;
	}
	
	public int value() {
		return value;
	}
}
