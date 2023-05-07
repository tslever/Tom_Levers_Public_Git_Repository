package com.tsl.inheritance__more_people;


/**
 * FullTimeFaculty represents the structure for a full-time faculty member.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
public class FullTimeFaculty extends Faculty implements Payroll {

	
	/**
	 * salary is an attribute of FullTimeFaculty.
	 */
	private double salary;
	
	
	/**
	 * FullTimeFaculty() is a zero-argument constructor for FullTimeFaculty that calls the three-argument constructor of
	 * Faculty with arguments "", 0, and "" and initializes salary to 0.00.
	 */
	public FullTimeFaculty() {
		//this.name = "";
		//this.empID = 0;
		//this.rank = "";
		super("", 0, "");
		this.salary = 0.00;
	}
	
	
	/**
	 * FullTimeFaculty(String name, int empID, String rank, double salary) is a four-argument constructor for
	 * FullTimeFaculty that calls the three-argument constructor of Faculty with arguments name, empID, and rank and
	 * initializes salary to theSalaryToUse.
	 * @param name
	 * @param empID
	 * @param rank
	 * @param theSalaryToUse
	 */
	public FullTimeFaculty(String name, int empID, String rank, double theSalaryToUse) {
		//this.name = name;
		//this.empID = empID;
		//this.salary = theSalaryToUse;
		super(name, empID, rank);
		this.salary = theSalaryToUse;
	}
	
	
	/**
	 * getTheSalary provides this faculty member's salary.
	 * @return
	 */
	public double getTheSalary() {
		return this.salary;
	}
	
	
	/**
	 * setTheSalary sets this faculty member's salary.
	 * @param theSalaryToUse
	 */
	public void setTheSalary(double theSalaryToUse) {
		this.salary = theSalaryToUse;
	}
	
	
	/**
	 * calculateMonthlyGrossPay() overrides and sufficiently implements calculateMonthlyGrossPay() insufficiently
	 * implemented in Person, and sufficiently implements the appropriate requirements in Payroll.
	 */
	@Override
	public double calculateMonthlyGrossPay() {
		
		return
			this.salary /*dollars per year*/ *
			1.0 / 52.0 /*years per week*/ *
			4.0 /*weeks per month*/
		;
		
	}
	
	
	/**
	 * calculateMonthlyGrossPay(int weeklyHours) overrides and sufficiently implements
	 * calculateMonthlyGrossPay(int weeklyHours) insufficiently implemented in Person, and sufficiently implements the
	 * appropriate requirements in Payroll.
	 */
	@Override
	public double calculateMonthlyGrossPay(int placeholder) {
		
		return
			this.salary /*dollars per year*/ *
			1.0 / 52.0 /*years per week*/ *
			4.0 /*weeks per month*/
		;
		
	}
	
	
	/**
	 * displayPayInfo() overrides and sufficiently implements displayPayInfo() insufficiently implemented in Person,
	 * and sufficiently implements the appropriate requirements in Payroll.
	 */
	@Override
	public void displayPayInfo() {
		
		System.out.println(
			"Pay is done on a yearly basis.\n" +
			"Salary: " + this.salary + "\n" +
			"Monthly salary: " + calculateMonthlyGrossPay(0) + "\n"
		);
		
	}
	
}