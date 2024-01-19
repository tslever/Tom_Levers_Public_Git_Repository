package brss.type_definer_and_assigner;

public class TemperatureValueLessThanLowerLimitException extends Exception {
	public TemperatureValueLessThanLowerLimitException(String errorMessage) {
		super(errorMessage);
	}
}
