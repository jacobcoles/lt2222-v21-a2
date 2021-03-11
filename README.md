# LT2222 V21 Assignment 2

Assignment 2 from the course LT2222 in the University of Gothenburg's winter 2021 semester.

Your name: Jacob Coles

## Part 1
I used the WordNetLemmatizer from NLTK to lemmatise the words. I didn't remove any words, as I'm sure the learning algorithm can handle it!

## Part 2

Will use start tokens (<s1>,<s2>,<s3>,<s4>,<s5>) and end tokens (<e1>,<e2>,<e3>,<e4>,<e5>).
These will hopefully encode the distance from the start and end of sentence. 
We will also include and go past other named entities (NEs). I have decided that the actual names of other neighbouring entities themselves aren't relevant, so will have a generic <ne> tag for these. The <ne> tag can represent a named entity spanning more than one token. 

## Part 3
Nothing interesting here 👁👄👁

## Part 4
🔥

## Part 5
I made the confusion matrices by first generating a list of all the classes, then I made a nested dictionary, to be able to save the values of truth vs. prediction. I then used these to generate the the confusion matrix/table. This is all done in the confusion_matrix() function. 
I also made a info_associated_w_cf() that just calculates metrics like true positive, false positive etc.

## Bonus 1
The worst performing classes are 'eve', 'art' and 'nat'. There are a few possible reasons in combination that contribute to this. The first is that we haven't preserved or encoded word order whatsoever, such that we can only guess a class by what words are around a NE. Additionally, these classes have so few examples compared to the other classes, so the system won't be as good at classifying these, and as such random chance (along with the aforementioned information loss), will cause another class to be selected. The other classes in general have a higher likelihood of being estimated given some random input. Setting aside the statistical information and information loss, there is only so much information to be retreived by the surrounding words, and some different classes may show up in situations where the context is nearly the same. If this type of situation is encountered, the one with more samples is likely to be chosen. 

## Bonus 2
I performed two methods. One of them simply added the pos tags for the contextual words to the list of features, whereas the other method simply calculated classes based on either pos tags OR word tags. I assume the latter method might be better if there were some way to combine the two sets as a final stage, but I'm unsure of how this would be done. 