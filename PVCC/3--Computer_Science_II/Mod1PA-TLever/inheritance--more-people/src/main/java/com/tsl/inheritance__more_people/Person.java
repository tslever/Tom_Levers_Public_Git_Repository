package com.tsl.inheritance__more_people;


/**
 * Person represents the structure for a person.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
class Person implements Payroll {

	private String theField;
	
	private class ATestClass {
		public ATestClass() {
			
		}
	}
	
	/**
	 * name is an attribute of a person.
	 */
	protected String name;
	
	
	/**
	 * Person() is a zero-argument constructor for Person that initializes a person's name to "".
	 */
	public Person() {
		this.name = "";
	}
	
	
	/**
	 * Person(String name) is a one-argument constructor for Person that initializes a person's name to name.
	 * @param name
	 */
	public Person(String name) {
		this.name = name;
	}
	
	
	/**
	 * getName provides the name of this person.
	 * @return
	 */
	public String getName() {
		return this.name;
	}
	
	
	/**
	 * setName sets the name of this person to name.
	 * @param name
	 */
	public void setName(String name) {
		this.name = name;
	}
	
	
	/**
	 * toString provides information for this person.
	 */
	@Override
	public String toString() {
		return "Name: " + this.name;
	}
	
	
	/**
	 * calculateMonthlyGrossPay() insufficiently implements the inherited abstract method calculateMonthlyGrossPay().
	 */
	public double calculateMonthlyGrossPay() throws ANotSufficientlyImplementedException { 
		throw new ANotSufficientlyImplementedException(
			"calculateMonthlyGrossPay() has not yet been implemented in a subclass of Person."
		);
	};

	
	/**
	 * calculateMonthlyGrossPay(int weeklyHoursOrCreditsPerSemester) insufficiently implements the inherited abstract
	 * method calculateMonthlyGrossPay(int weeklyHoursOrCreditsPerSemester).
	 */
	public double calculateMonthlyGrossPay(int weeklyHoursPlaceholderOrCreditsPerSemester)
		throws ANotSufficientlyImplementedException { 
		throw new ANotSufficientlyImplementedException(
			"calculateMonthlyGrossPay(int weeklyHoursPlaceholderOrCreditsPerSemester) has not yet been implemented " +
			"in a subclass of Person."
		);
	};
	

	/**
	 * displayPayInfo implements the inherited abstract method displayPayInfo.
	 */
	public void displayPayInfo() throws ANotSufficientlyImplementedException {
		throw new ANotSufficientlyImplementedException(
			"displayPayInfo() has not yet been implemented in a subclass of Person."
		);
	}
	
	
	private void test() {
		
	}
	
}