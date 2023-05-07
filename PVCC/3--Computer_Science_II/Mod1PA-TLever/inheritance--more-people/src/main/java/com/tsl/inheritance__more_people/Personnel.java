package com.tsl.inheritance__more_people;

/**
 * Hello world!
 *
 */
class Personnel 
{
    public static void main( String[] args )
    {
        GTA p1 = new GTA("Anna Smiley", 23234, "Science", 20.00);
        
        FullTimeFaculty p2 = new FullTimeFaculty("Jane Dane", 2343, "Lecturer", 49000.00);
        
        AdjunctFaculty p3 = new AdjunctFaculty("Edward Stone", 121, "Assistant Professor", 950.00);
        
        System.out.println("Graduate Teaching Assistant Information:\n");
        System.out.println(p1);
        p1.displayPayInfo();
        
        System.out.println("Full time faculty information:\n");
        System.out.println(p2);
        p2.displayPayInfo();
        
        System.out.println("Adjunct faculty information:\n");
        System.out.println(p3);
        p3.displayPayInfo();
        
    }
}
