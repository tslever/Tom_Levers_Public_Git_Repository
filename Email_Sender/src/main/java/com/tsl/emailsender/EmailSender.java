package com.tsl.emailsender;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.reflect.TypeToken;

import jakarta.activation.DataSource;
import jakarta.activation.FileDataSource;

/**
 * An {@code EmailSender} object represents an email sender.
 */
public abstract class EmailSender {

    Configurations configurations;
    EmailInfo emailInfo;
   
    /**
     * Constructs an email sender based on a {@code Configurations} object and an {@code EmailInfo} object.
     * 
     * @param configurationsToUse a {@code Configurations} object with which to construct an email sender
     * @param emailInfoToUse an {@code EmailInfo} object with which to construct an email sender
     * @throws IOException if email information cannot be gotten
     */
    public EmailSender(Configurations configurationsToUse, EmailInfo emailInfoToUse) throws IOException {
        configurations = configurationsToUse;
        emailInfo = emailInfoToUse;
        if (emailInfoToUse == null) {
            emailInfo = getEmailInfo();
        }
    }
    
    /**
     * Returns an {@code EmailInfo} object with a from address, a list of to addresses, a subject, a body, and a list of attachments.
     * 
     * @return an {@code EmailInfo} object with a from address, a list of to addresses, a subject, a body, and a list of attachments
     * @throws IOException if an email header or body cannot be read
     */
    public EmailInfo getEmailInfo() throws IOException {
        Path pathToHeader = Path.of(configurations.getEmailDirectory() + "/header.json");
        String headerAsString = Files.readString(pathToHeader, StandardCharsets.UTF_8);
        JsonObject header = JsonParser.parseString(headerAsString).getAsJsonObject();
        String fromAddress = header.get("from").getAsString();
        JsonArray jsonArrayOfToAddresses = header.get("to").getAsJsonArray();
        ArrayList<String> arrayListOfToAddresses = (new Gson()).fromJson(jsonArrayOfToAddresses, new TypeToken<ArrayList<String>>(){}.getType());
        String subject = header.get("subject").getAsString();
        Path pathToBody = Path.of(configurations.getEmailDirectory() + "/body.txt");
        String body = Files.readString(pathToBody, StandardCharsets.UTF_8);
        File directoryForAttachments = new File(configurations.getEmailDirectory() + "/attachments");
        ArrayList<DataSource> arrayListOfAttachments = new ArrayList<>();
        for (File file : directoryForAttachments.listFiles()) {
            if (!file.isDirectory()) {
                FileDataSource attachment = new FileDataSource(file);
                arrayListOfAttachments.add(attachment);
            }
        }
        return new EmailInfo(fromAddress, arrayListOfToAddresses, subject, body, arrayListOfAttachments);
    }
    
    
    /**
     * Sends an email.
     * 
     * @throws Exception if this email sender cannot send an email
     */
    public abstract void send() throws Exception;
}