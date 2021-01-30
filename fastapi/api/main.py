from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app=FastAPI()

fakedb = []

class Course(BaseModel):
	id: int
	name: str
	price: float
	is_early_bird: Optional[bool] = None


@app.get("/")
def read_root():
	return {"greetings": "Hello world!"}


@app.get('/courses')
def get_courses():
	return fakedb


@app.get('/courses/{course_id}')
def get_courses(course_id: int):
	return fakedb[course_id-1]


@app.post('courses')
def add_course(course: Course):
	fakedb.append(course.dict())
	return fakedb[-1]


@app.delete('courses/{course_id}')
def delete_course(course_id: int):
	fakedb.pop(course_id-1)
	return {"task": "deletion successful"}