package com.tsl.emailsender;

import java.util.ArrayList;

import jakarta.activation.FileDataSource;

/**
 * An {@code EmailInfo} object encapsulates email information including a from address, a list of to addresses, a subject, a body, and a list of attachments.
 */
public class EmailInfo {
    
    private String fromAddress;
    private ArrayList<String> arrayListOfToAddresses;
    private String subject;
    private String body;
    private ArrayList<FileDataSource> arrayListOfAttachments;

    /**
     * Constructs an {@code EmailInfo} object with a from address, a list of to addresses, a subject, a body, and a list of attachments.
     * 
     * @param fromAddressToUse a from address
     * @param listOfToAddressesToUse a list of to addresses
     * @param subjectToUse a subject
     * @param bodyToUse a body
     * @param listOfAttachmentsToUse a list of attachments
     */
    public EmailInfo(String fromAddressToUse, ArrayList<String> listOfToAddressesToUse, String subjectToUse, String bodyToUse, ArrayList<FileDataSource> listOfAttachmentsToUse) {
        fromAddress = fromAddressToUse;
        arrayListOfToAddresses = listOfToAddressesToUse;
        subject = subjectToUse;
        body = bodyToUse;
        arrayListOfAttachments = listOfAttachmentsToUse;
    }
    
    /**
     * Returns the from address of this {@code EmailInfo} object.
     * 
     * @return the from address of this {@code EmailInfo} object
     */
    public String getFromAddress() {
        return fromAddress;
    }

    /**
     * Returns the list of to addresses of this {@code EmailInfo} object.
     * 
     * @return the list of to addresses of this {@code EmailInfo} object
     */
    public ArrayList<String> getListOfToAddresses() {
        return arrayListOfToAddresses;
    }
    
    /**
     * Returns the subject of this {@code EmailInfo} object.
     * 
     * @return the subject of this {@code EmailInfo} object
     */
    public String getSubject() {
        return subject;
    }
    
    /**
     * Returns the body of this {@code EmailInfo} object.
     * 
     * @return the body of this {@code EmailInfo} object
     */
    public String getBody() {
        return body;
    }
    
    /**
     * Returns the list of attachments of this {@code EmailInfo} object.
     * 
     * @return the list of attachments of this {@code EmailInfo} object
     */
    public ArrayList<FileDataSource> getListOfAttachments() {
        return arrayListOfAttachments;
    }
}