package com.tsl.emailsender;

import java.io.IOException;
import java.security.GeneralSecurityException;

import javax.mail.MessagingException;

/**
 * Encapsulates a main method, which creates and configures an email sender and has that sender send an email.
 *
 * @author Tom Lever
 * @version 0.0
 * @since 08/11/2022
 */
public class Main {
    
    /**
     * Creates and configures an email sender and has that sender send an email.
     *
     * @param args, including:
     *     - path to directory of credentials (e.g., /Users/tlever/CNRI_Projects/email-sender/credentials)
     *     - path to directory of email data (e.g., /Users/tlever/CNRI_Projects/email-sender/emails)
     *     - email ID (e.g., monthlyReportOnHandlePrefixRegistrationsAndRenewals)
     * @throws GeneralSecurityException
     * @throws IOException
     * @throws MessagingException if an Internet address, MIME message, or byte array output stream cannot be fully assembled.
     */
    public static void main(String[] args) throws GeneralSecurityException, IOException, MessagingException {
        
    	String credentialsDirectory = args[0];
    	String emailDirectory = args[1];
        String email = args[2];
        
        Configurations configurations = new Configurations(credentialsDirectory, email, emailDirectory);
        EmailSender emailSender = new EmailSender(configurations);
        emailSender.send();
    }
}