package com.tsl.inheritance__more_people;



/**
 * Payroll represents a template that implementing classes must fit; Payroll describes requirements for implementing
 * classes.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
abstract interface Payroll {

	String theField = "";
	
	abstract class ATestClass { };
	
	/**
	 * calculateMonthlyGrossPay() requires implementing classes to implement calculateMonthlyGrossPay().
	 * calculateMonthlyGrossPay() will, for Graduate Teaching Assistants, calculate monthly gross
	 * pay on the basis of an hourly salary, 40.0 hours per week of work, and 4.0 weeks per month.
	 * calculateMonthlyGrossPay() will, for full-time faculty, calculate monthly gross
	 * pay on the basis of a yearly salary, 52.0 weeks per year, and 4.0 weeks per month.
	 * calculateMonthlyGrossPay() will, for adjunct faculty, calculate monthly gross pay on the basis of a per-credit
	 * rate, 15.0 credits per semester, and 2.0 months per semester.
	 * @return
	 * @throws ANotSufficientlyImplementedException
	 */
	abstract double calculateMonthlyGrossPay() throws ANotSufficientlyImplementedException;

	
	/**
	 * calculateMonthlyGrossPay() requires implementing classes to implement calculateMonthlyGrossPay().
	 * calculateMonthlyGrossPay() will, for Graduate Teaching Assistants, calculate monthly gross
	 * pay on the basis of an hourly salary, 40.0 hours per week of work, and 4.0 week per month.
	 * calculateMonthlyGrossPay() will, for full-time faculty, calculate monthly gross
	 * pay on the basis of a yearly salary, 52.0 weeks per year, and 4.0 weeks per month.
	 * calculateMonthlyGrossPay() will, for adjunct faculty, calculate monthly gross pay on the basis of a per-credit
	 * rate, 15.0 credits per semester, and 2.0 months per semester.
	 * @return
	 * @throws ANotSufficientlyImplementedException
	 */
	abstract double calculateMonthlyGrossPay(int weeklyHoursPlaceholderOrCreditsPerSemester)
		throws ANotSufficientlyImplementedException;
	
	
	/**
	 * displayPayInfo requires implementing classes to implement displayPayInfo.
	 * @throws ANotSufficientlyImplementedException
	 */
	abstract void displayPayInfo() throws ANotSufficientlyImplementedException;
	
}
