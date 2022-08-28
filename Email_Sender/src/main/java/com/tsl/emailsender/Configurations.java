package com.tsl.emailsender;

public class Configurations {
    
    public String credentialsDirectory;
    public String email;
    public String emailsDirectory;
    
    public Configurations(String credentialsDirectoryToUse, String emailToUse, String emailsDirectoryToUse) {
        credentialsDirectory = credentialsDirectoryToUse;
        email = emailToUse;
        emailsDirectory = emailsDirectoryToUse;
    }
}