package Com.TSL.LineEditorUtilities;


/**
 * ACommand represents the structure for a command related to file-manipulation and/or line-editing functionality.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class ACommand implements Comparable<ACommand> {

	
	private String name;
	private String text;
	private AnEncapsulatorForEdit encapsulatorForEdit;
	
	
	/**
	 * ACommand(String theNameToUse, String theTextToUse) is the two-parameter constructor for ACommand, which sets this
	 * command's name to theNameToUse and sets this command's text to the text to use.
	 * 
	 * @param theNameToUse
	 * @param theCommandToUse
	 */
	
	public ACommand(String theNameToUse, String theTextToUse, AnEncapsulatorForEdit theEncapsulatorForEditToUse) {
		
		this.name = theNameToUse;
		this.text = theTextToUse;
		this.encapsulatorForEdit = theEncapsulatorForEditToUse; 
		
	}
	
	
	/**
	 * compareTo indicates whether a provided command has a name that is less than, equal to, or greater than the name
	 * of this command.
	 */
	
	@Override
	public int compareTo(ACommand theCommand) {
		
		return this.name.compareTo(theCommand.name);
		
	}
	
	
	/**
	 * equals indicates whether or not a provided object has the same name and text as this command.
	 */
	
	@Override
	public boolean equals(Object theObject) {
		
		if (theObject == this) {
			return true;
		}
		
		if (theObject == null || theObject.getClass() != this.getClass()) {
			return false;
		}
		
		ACommand theCommand = (ACommand)theObject;
		
		return ((this.name.equals(theCommand.name)) && (this.text.equals(theCommand.text)));
		
	}
	
	
	/**
	 * providesItsEncapsulatorForEdit provides the encapsulator for a file-manipulation and/or line-editing method
	 * associated with this command.
	 * 
	 * @return
	 */
	
	public AnEncapsulatorForEdit providesItsEncapsulatorForEdit() {
		return this.encapsulatorForEdit;
	}
	
	
	/**
	 * toString provides a string representation of this command.
	 */
	
	@Override
	public String toString() {
		
		return this.name + ": " + this.text;
		
	}
	
}
