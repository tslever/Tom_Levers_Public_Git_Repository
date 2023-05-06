package com.tsl.person_class;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-07
*
* Student name:  Tom Lever
* Completion date: 05/17/21
* 
* Person represents the structure for a person.
*/

class Person
{
	//*** Task #1: define the instance variables
	
	/**
	 * name is an attribute of a person.
	 */
	private String name;
	
	/**
	 * idNumber is an attribute of a person.
	 */
	private int idNumber;
	
	/**
	 * department is an attribute of a person.
	 */
	private String department;
	

	//*** Task #2: write the no-argument constructor
	
	/**
	 * Person() is a zero-argument constructor for Person that initializes a person's name to "", the person's
	 * idNumber to 0, and the person's department to "".
	 */
	public Person() {
		this.name = "";
		this.idNumber = 0;
		this.department = "";
	}
	

	//*** Task #3: write the constructor that passes values for the name and idNumber
	
	/**
	 * Person(String theNameToUse, int theIdNumberToUse) is a two-argument constructor that initializes a person's
	 * name to theNameToUse, the person's idNumber to theIdNumberToUse, and the person's department to "".
	 * @param theNameToUse
	 * @param theIdNumberToUse
	 */
	public Person(String theNameToUse, int theIdNumberToUse) {
		this.name = theNameToUse;
		this.idNumber = theIdNumberToUse;
		this.department = "";
	}
	
	
	//*** Task #4: write the constructor that initializes all three instance variables
	
	/**
	 * Person(String theNameToUse, int theIdNumberToUse, String theDepartmentToUse) is a three-argument constructor
	 * that initializes a person's name to theNameToUse, the person's idNumber to theIdNumberToUse, and the person's
	 * department to theDepartmentToUse.
	 * @param theNameToUse
	 * @param theIdNumberToUse
	 * @param theDepartmentToUse
	 */
	public Person(String theNameToUse, int theIdNumberToUse, String theDepartmentToUse) {
		this.name = theNameToUse;
		this.idNumber = theIdNumberToUse;
		this.department = theDepartmentToUse;
	}
	
	
	//*** Task #5: write accessor method for attribute name
	
	/**
	 * getName provides the name of this person.
	 * @return
	 */
	public String getName() {
		return this.name;
	}
	
	
	//*** Task #6: write mutator method for attribute name
	
	/**
	 * setName sets the name of this person to a provided name.
	 * @param theNameToUse
	 */
	public void setName(String theNameToUse) {
		this.name = theNameToUse;
	}
	

	//*** Task #7: write accessor method for attribute idNumber
	
	/**
	 * getIdNumber provides the ID number of this person.
	 * @return
	 */
	public int getIdNumber() {
		return this.idNumber;
	}
	
	
	//*** Task #8: write mutator method for attribute idNumber
	
	/**
	 * setIdNumber sets the ID number of this person to a provided ID number.
	 * @param theIdNumberToUse
	 */
	public void setIdNumber(int theIdNumberToUse) {
		this.idNumber = theIdNumberToUse;
	}
	
	
	//*** Task #9: write accessor method for attribute department
	
	/**
	 * getDepartment provides the department of this person.
	 * @return
	 */
	public String getDepartment() {
		return this.department;
	}
	
	
	//*** Task #10: write mutator method for attribute department
	
	/**
	 * setDepartment sets the department of this person to a provided department.
	 * @param theDepartmentToUse
	 */
	public void setDepartment(String theDepartmentToUse) {
		this.department = theDepartmentToUse;
	}
	
	
	//*** Task #11: write toString method
	
	/**
	 * toString outputs information for this person.
	 */
	@Override
	public String toString() {
		return
			"Name: " + this.name + "\n" +
			"ID number: " + this.idNumber + "\n" +
			"Department: " + this.department;
	}
	
}
