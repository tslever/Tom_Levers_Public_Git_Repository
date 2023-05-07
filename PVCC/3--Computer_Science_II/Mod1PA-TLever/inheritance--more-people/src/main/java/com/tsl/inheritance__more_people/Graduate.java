package com.tsl.inheritance__more_people;


/**
 * Graduate represents the structure for a graduate student.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
class Graduate extends Student implements Payroll {

	
	/**
	 * department is an attribute of Graduate.
	 */
	protected String department;
	
	
	/**
	 * Graduate() is a zero-argument constructor for Graduate that calls the two-argument constructor of Student with
	 * arguments "" and 0 and initializes department to "".
	 */
	public Graduate() {
		//this.name = "";
		//this.studentNumber = 0;
		super("", 0);
		this.department = "";
	}
	
	
	/**
	 * Graduate(String name, int studentNumber, String department) is a three-argument constructor for Graduate that
	 * calls the two-argument constructor of Student with arguments name and studentNumber and initializes department
	 * to theDepartmentToUse.
	 * @param name
	 * @param studentNumber
	 * @param major
	 */
	public Graduate(String name, int studentNumber, String theDepartmentToUse) {
		//this.name = name;
		//this.studentNumber = studentNumber;
		super(name, studentNumber);
		this.department = theDepartmentToUse;
	}
	
	
	/**
	 * getDepartment provides this graduate's department.
	 * @return
	 */
	public String getTheDepartment() {
		return this.department;
	}
	
	
	/**
	 * setDepartment sets this graduate's department.
	 * @param theDepartmentToUse
	 */
	public void setTheDepartment(String theDepartmentToUse) {
		this.department = theDepartmentToUse;
	}
	
	
	/**
	 * toString provides information for this graduate.
	 */
	@Override
	public String toString() {
		return
			"Name: " + this.name + "\n" +
			"Student Number: " + this.studentNumber + "\n" +
			"Graduate department: " + this.department;
	}
	
	
}
