package com.tsl.emailsender;

import java.io.File;
import java.io.IOException;
import java.security.GeneralSecurityException;
import java.util.Arrays;
import java.util.List;

import jakarta.activation.DataSource;
import jakarta.activation.FileDataSource;
import jakarta.mail.MessagingException;

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
     * @param args including:
     *     - path to directory of server configuration (e.g., /Users/tlever/CNRI_Projects/email-sender/server-configurations/deepsouth.cnri.net-465)
     *     - path to directory of credentials (e.g., /Users/tlever/CNRI_Projects/email-sender/credentials/tlever@cnri.reston.va.us)
     *     - path to directory of email data (e.g., /Users/tlever/CNRI_Projects/email-sender/emails/monthlyReportOnHandlePrefixRegistrationsAndRenewals)
     * @throws GeneralSecurityException if an email cannot be sent using Gmail
     * @throws IOException if an email cannot be sent using Simple Java Mail or Gmail
     * @throws MessagingException if an email cannot be sent using Gmail
     */
    public static void main(String[] args) throws GeneralSecurityException, IOException, MessagingException {
    	String serverConfigurationDirectory = args[0];
    	String credentialsDirectory = args[1];
    	
//        String emailDirectory = args[2];
//        Configurations configurations = new Configurations(serverConfigurationDirectory, credentialsDirectory, emailDirectory);
//        SimpleJavaMailSender simpleJavaMailSender = new SimpleJavaMailSender(configurations, null);
        
        Configurations configurations = new Configurations(serverConfigurationDirectory, credentialsDirectory, null);
        File file = new File("/Users/tlever/Desktop/file.txt");
        FileDataSource fileDataSource = new FileDataSource(file); 
        List<String> listOfToAddresses = Arrays.asList("tlever@cnri.reston.va.us");
        List<DataSource> listOfAttachments = Arrays.asList(fileDataSource);
        EmailInfo emailInfo = new EmailInfo("test@test.com", listOfToAddresses, "Test Subject", "Test Body", listOfAttachments);
        SimpleJavaMailSender simpleJavaMailSender = new SimpleJavaMailSender(configurations, emailInfo);
        
        simpleJavaMailSender.send();
    }
}