package com.tsl.inheritance__more_people;


/**
 * GTA represents the structure for a Graduate Teaching Assistant.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
public class GTA extends Graduate implements Payroll {

	
	/**
	 * hourlySalary is an attribute of GTA.
	 */
	private double hourlySalary;
	
	
	/**
	 * GTA() is a zero-argument constructor for GTA that calls the three-argument constructor of Graduate with
	 * arguments "", 0, and "" and initializes hourly salary to 0.00.
	 */
	public GTA() {
		//this.name = "";
		//this.studentNumber = 0;
		//this.department = "";
		super("", 0, "");
		this.hourlySalary = 0.00;
	}
	
	
	/**
	 * GTA(String name, int studentNumber, String department, double hourlySalary) is a four-argument constructor for
	 * GTA that calls the three-argument constructor of Graduate with arguments name, studentNumber, and department and
	 * initializes hourlySalary to theHourlySalaryToUse.
	 * @param name
	 * @param studentNumber
	 * @param department
	 * @param theHourlySalaryToUse
	 */
	public GTA(String name, int studentNumber, String theDepartmentToUse, double theHourlySalaryToUse) {
		//this.name = name;
		//this.studentNumber = studentNumber;
		//this.hourlySalary = theHourlySalaryToUse;
		super(name, studentNumber, theDepartmentToUse);
		this.hourlySalary = theHourlySalaryToUse;
	}
	
	
	/**
	 * getTheHourlySalary provides this GTA's hourly salary.
	 * @return
	 */
	public double getTheHourlySalary() {
		return this.hourlySalary;
	}
	
	
	/**
	 * setTheHourlySalary sets this GTA's hourly salary.
	 * @param theHourlySalaryToUse
	 */
	public void setTheHourlySalary(double theHourlySalaryToUse) {
		this.hourlySalary = theHourlySalaryToUse;
	}
	
	
	/**
	 * calculateMonthlyGrossPay() overrides and sufficiently implements calculateMonthlyGrossPay() insufficiently
	 * implemented in Person, and sufficiently implements the appropriate requirements in Payroll.
	 */
	@Override
	public double calculateMonthlyGrossPay() {
		
		return
			this.hourlySalary /*dollars per hour*/ *
			40.0 /*hours per week*/ *
			4.0 /*weeks per month*/
		;
		
	}
	
	
	/**
	 * calculateMonthlyGrossPay(int weeklyHours) overrides and sufficiently implements
	 * calculateMonthlyGrossPay(int weeklyHours) insufficiently implemented in Person, and sufficiently implements the
	 * appropriate requirements in Payroll.
	 */
	@Override
	public double calculateMonthlyGrossPay(int weeklyHours) {
		
		return
			this.hourlySalary /*dollars per hour*/ *
			(double)weeklyHours /*hours per week*/ *
			4.0 /*weeks per month*/
		;
		
	}
	
	
	/**
	 * displayPayInfo() overrides and sufficiently implements displayPayInfo() insufficiently implemented in Person,
	 * and sufficiently implements the appropriate requirements in Payroll.
	 */
	@Override
	public void displayPayInfo() {
		
		int weeklyHours = 40;
		
		System.out.println(
			"Pay is done on an hourly basis.\n" +
			"Number of hours per week: " + weeklyHours + "\n" +
			"Pay per hour: " + this.hourlySalary + "\n" +
			"Monthly salary: " + calculateMonthlyGrossPay(weeklyHours) + "\n"
		);
		
	}
	
	
}
