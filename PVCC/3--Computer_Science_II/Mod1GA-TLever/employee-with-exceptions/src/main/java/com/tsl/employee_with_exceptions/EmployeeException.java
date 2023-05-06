package com.tsl.employee_with_exceptions;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-07
*
* Student name:  Tom Lever
* Completion date: 05/18/21
*
* This class represents the blueprint for instantiating EmployeeException objects,
* with the following attributes:

	name: String
	SSN: String
	salary: double
* and methods:
  	A constructor with no arguments that sets the attributes at default values
	A constructor that passes values for all attributes
	Accessor, mutator and display methods for each attribute
	An equals method that has an object of type Employee as argument, and returns true if two employees have the same name, salary and sSN
*/

class EmployeeException {
	
	/**
	 * name is an attribute of an employee.
	 */
	private String name;
	
	
	/**
	 * SSN is an attribute of an employee.
	 */
	private String SSN;
	
	
	/**
	 * Salary is an attribute of an employee.
	 */
	private double Salary;
	
	
	/**
	 * EmployeeException is a zero-argument constructor for EmployeeException that initializes name to "", SSN to "",
	 * and Salary to 0.
	 */
	public EmployeeException() {
		this.name = "";
		this.SSN = "";
		this.Salary = 0;
	}
	
	
	/**
	 * EmployeeException(String theNameToUse, String theSocialSecurityNumberToUse, double theSalaryToUse) is a three-
	 * argument constructor for EmployeeException that initializes name to theNameToUse, SSN to
	 * theSocialSecurityNumberToUse, and Salary to theSalaryToUse.
	 * @param theNameToUse
	 * @param theSocialSecurityNumberToUse
	 * @param theSalaryToUse
	 */
	public EmployeeException(String theNameToUse, String theSocialSecurityNumberToUse, double theSalaryToUse) {
		this.name = theNameToUse;
		this.SSN = theSocialSecurityNumberToUse;
		this.Salary = theSalaryToUse;
	}
	
	
	/**
	 * getName provides this employee's name.
	 * @return
	 */
	public String getName() {
		return this.name;
	}
	
	
	/**
	 * getSSN provides this employee's Social Security Number.
	 * @return
	 */
	public String getSSN() {
		return this.SSN;
	}
	
	
	/**
	 * getSalary provides this employee's salary.
	 * @return
	 */
	public double getSalary() {
		return this.Salary;
	}
	
	
	/**
	 * setName sets this employee's name.
	 * @param theNameToUse
	 */
	public void setName(String theNameToUse) {
		this.name = theNameToUse;
	}
	
	
	/**
	 * setSSN sets this employee's Social Security Number.
	 * @param theSocialSecurityNumberToUse
	 */
	public void setSSN(String theSocialSecurityNumberToUse) {
		this.SSN = theSocialSecurityNumberToUse;
	}
	
	
	/**
	 * setSalary sets this employee's salary.
	 * @param theSalaryToUse
	 */
	public void setSalary(double theSalaryToUse) {
		this.Salary = theSalaryToUse;
	}
	
	
	/**
	 * writeOutName displays information relating to this employee's name.
	 */
	public void writeOutName() {
		System.out.println("Employee Name: " + this.name);
	}
	
	
	/**
	 * format provides a version of this employee's Social Security Number with hyphens.
	 * @param theSocialSecurityNumber
	 * @return
	 */
	public String format(String theSocialSecurityNumber) {
		return
			this.SSN.charAt(0) +
			this.SSN.charAt(1) +
			this.SSN.charAt(2) +
			"-" +
			this.SSN.charAt(3) +
			this.SSN.charAt(4) +
			"-" +
			this.SSN.charAt(5) +
			this.SSN.charAt(6) +
			this.SSN.charAt(7) +
			this.SSN.charAt(8);
	}
	
	
	/**
	 * writeOutSSN displays information relating to this employee's Social Security Number.
	 */
	public void writeOutSSN() {
		System.out.println("Employee SSN: " + format(this.SSN));
	}
	
	
	/**
	 * writeOutSalary displays information relating to this employee's salary.
	 */
	public void writeOutSalary() {
		System.out.println("Employee salary: " + this.Salary);
	}
	
	
	/**
	 * writeOutput displays information relating to this employee.
	 */
	public void writeOutput() {
		System.out.println(
			"Employee Name: " + this.name + "\n" +
			"Employee SSN: " + format(this.SSN) + "\n" + 
			"Employee salary: " + this.Salary
		);
	}
	
	
	/**
	 * equals provides information regarding whether all the attributes of this employee are equal to the corresponding
	 * attributes of another employee or not.
	 * @param theEmployeeExceptionToUse
	 * @return
	 */
	public boolean equals(EmployeeException theEmployeeExceptionToUse) {
		return (
			(this.name == theEmployeeExceptionToUse.name) &&
			(this.SSN == theEmployeeExceptionToUse.SSN) &&
			(this.Salary == theEmployeeExceptionToUse.Salary)
		);
	}
	
}