package wav_assembling_utilities;

public class StartTickHasNoCorrespondingEndTickException extends Exception {
	
	public StartTickHasNoCorrespondingEndTickException(String errorMessage) {
		super(errorMessage);
	}
}