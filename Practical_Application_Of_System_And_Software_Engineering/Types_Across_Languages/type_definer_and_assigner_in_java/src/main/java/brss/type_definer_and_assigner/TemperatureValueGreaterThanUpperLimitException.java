package brss.type_definer_and_assigner;

public class TemperatureValueGreaterThanUpperLimitException extends Exception {
	public TemperatureValueGreaterThanUpperLimitException(String errorMessage) {
		super(errorMessage);
	}
}
