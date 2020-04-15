class Palindrome:
  @staticmethod
  def is_palindrome(word):
      reverse = word[::-1]
      if reverse.upper() == word.upper():
          return True
      else:
          return False
word = input()
print(Palindrome.is_palindrome(word))