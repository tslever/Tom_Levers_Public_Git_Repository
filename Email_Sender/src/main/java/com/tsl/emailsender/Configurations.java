package com.tsl.emailsender;

/**
 * A {@code Configurations} object encapsulates configurations for getting credentials and email information.
 */
public class Configurations {
    
    private String serverConfigurationDirectory;
    private String credentialsDirectory;
    private String emailDirectory;
    
    /**
     * Constructs a {@code Configurations} object with server-configuration directory, credentials directory, and email directory.
     * 
     * @param serverConfigurationDirectoryToUse a server-configuration directory
     * @param credentialsDirectoryToUse a credentials directory
     * @param emailDirectoryToUse an email directory
     */
    public Configurations(String serverConfigurationDirectoryToUse, String credentialsDirectoryToUse, String emailDirectoryToUse) {
        serverConfigurationDirectory = serverConfigurationDirectoryToUse;
        credentialsDirectory = credentialsDirectoryToUse;
        emailDirectory = emailDirectoryToUse;
    }
    
    /**
     * Returns the server-configuration directory of this {@code Configurations} object.
     * 
     * @return the server-configuration directory of this {@code Configurations} object
     */
    public String getServerConfigurationDirectory() {
        return serverConfigurationDirectory;
    }
    
    /**
     * Returns the credentials directory of this {@Configurations} object.
     * 
     * @return the credentials directory of this {@code Configurations object}
     */
    public String getCredentialsDirectory() {
        return credentialsDirectory;
    }

    /**
     * Returns the path to an email of this {@Configurations} object.
     * 
     * @return the path to an email of this {@code Configurations object}
     */
    public String getEmailDirectory() {
        return emailDirectory;
    }
}