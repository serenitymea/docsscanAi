import json
from pathlib import Path

import docx
import PyPDF2
import pandas as pd
from bs4 import BeautifulSoup


class DocumentProcessor:
    """Document processor for various file formats"""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.html', '.htm', '.csv', '.xlsx', '.xls', '.json'}
    
    @staticmethod
    def extract_text_from_file(file_path: str) -> str:
        """Extract text from various file formats"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension not in DocumentProcessor.SUPPORTED_EXTENSIONS:
            raise ValueError(f"unsupported file format: {extension}")
        
        try:
            if extension == '.txt':
                return DocumentProcessor._extract_from_txt(file_path)
            elif extension == '.pdf':
                return DocumentProcessor._extract_from_pdf(file_path)
            elif extension == '.docx':
                return DocumentProcessor._extract_from_docx(file_path)
            elif extension in ['.html', '.htm']:
                return DocumentProcessor._extract_from_html(file_path)
            elif extension == '.csv':
                return DocumentProcessor._extract_from_csv(file_path)
            elif extension in ['.xlsx', '.xls']:
                return DocumentProcessor._extract_from_excel(file_path)
            elif extension == '.json':
                return DocumentProcessor._extract_from_json(file_path)
                
        except Exception as e:
            raise Exception(f"error processing file {file_path}: {str(e)}")
    
    @staticmethod
    def _extract_from_txt(file_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            for encoding in ['cp1251', 'latin1']:
                try:
                    return file_path.read_text(encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise Exception("unable to decode text file")
    
    @staticmethod
    def _extract_from_pdf(file_path: Path) -> str:
        """Extract text from PDF"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            raise Exception(f"error reading PDF: {str(e)}")
        
        if not text.strip():
            raise Exception("no text could be extracted from PDF")
        return text
    
    @staticmethod
    def _extract_from_docx(file_path: Path) -> str:
        """Extract text from DOCX"""
        try:
            doc = docx.Document(file_path)
            text = ""

            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"

            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text += " | ".join(row_text) + "\n"
            
            return text
        except Exception as e:
            raise Exception(f"error reading DOCX: {str(e)}")
    
    @staticmethod
    def _extract_from_html(file_path: Path) -> str:
        """Extract text from HTML"""
        try:
            html_content = file_path.read_text(encoding='utf-8')
            soup = BeautifulSoup(html_content, 'html.parser')

            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            raise Exception(f"Error reading HTML: {str(e)}")
    
    @staticmethod
    def _extract_from_csv(file_path: Path) -> str:
        """Extract text from CSV"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            return df.to_string(index=False)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='cp1251')
                return df.to_string(index=False)
            except Exception as e:
                raise Exception(f"error reading CSV: {str(e)}")
        except Exception as e:
            raise Exception(f"error reading CSV: {str(e)}")
    
    @staticmethod
    def _extract_from_excel(file_path: Path) -> str:
        """Extract text from Excel"""
        try:
            text = ""
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                text += f"Sheet: {sheet_name}\n{df.to_string(index=False)}\n\n"
                
            return text
        except Exception as e:
            raise Exception(f"error reading Excel: {str(e)}")
    
    @staticmethod
    def _extract_from_json(file_path: Path) -> str:
        """Extract text from JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            raise Exception(f"error reading JSON: {str(e)}")