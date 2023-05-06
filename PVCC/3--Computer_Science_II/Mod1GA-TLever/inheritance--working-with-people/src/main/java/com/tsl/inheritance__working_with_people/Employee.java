package com.tsl.inheritance__working_with_people;


/**
 * Employee represents the structure for an employee.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 *
 */
class Employee extends Person {

	
	/**
	 * empID is an attribute of an employee.
	 */
	protected int empID;
	
	
	/**
	 * Employee() is a zero-argument constructor for Employee that calls the one-argument constructor of Person with
	 * argument "" and initializes empID to 0.
	 */
	public Employee() {
		//this.name = "";
		super("");
		this.empID = 0;
	}
	
	
	/**
	 * Employee(String name, int empID) is a two-argument constructor for Employee that calls the one-argument
	 * constructor of Person with argument name and initializes empID to empID.
	 * @param name
	 * @param empID
	 */
	public Employee(String name, int empID) {
		//this.name = name;
		super(name);
		this.empID = empID;
	}
	
	
	/**
	 * getEmployeeID provides this employee's employee ID.
	 * @return
	 */
	public int getEmployeeID() {
		return this.empID;
	}
	
	
	/**
	 * setEmployeeID sets this employee's employee ID.
	 * @param employeeID
	 */
	public void setEmployeeID(int employeeID) {
		this.empID = employeeID;
	}
	
	
	/**
	 * toString provides information for this employee.
	 */
	@Override
	public String toString() {
		return
			"Name: " + this.name + "\n" +
			"Employee ID: " + this.empID;
	}
	
}
